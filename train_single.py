"""
Single-session sanity check: train on first 90% of one scan, test on last 10%.
Supports fixed-atlas mode (default) and learnable Gaussian bubble mode (--n_cortical).
"""

import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import nibabel as nib
import hcp_utils as hcp
from model import TimeSeriesTransformer
from bubble_parc import BubbleParcellation, bubble_regularizers


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",          default="data/100307_REST1_LR_p.npy")
    p.add_argument("--raw",           default="data/100307_REST1_LR_raw.npy")
    p.add_argument("--cifti",         default="data/100307_REST1_LR.dtseries.nii",
                   help="CIFTI file used to extract subcortical voxel coords (bubble mode only)")
    p.add_argument("--window",        type=int,   default=30)
    p.add_argument("--train_split",   type=float, default=0.9)
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--dim_val",       type=int,   default=760)
    p.add_argument("--n_heads",       type=int,   default=8)
    p.add_argument("--n_enc",         type=int,   default=4)
    p.add_argument("--n_dec",         type=int,   default=4)
    p.add_argument("--out",           default="data/single_session_model.pt")
    # fixed-atlas mode
    p.add_argument("--loss",          default="mc_raw", choices=["mmp", "mc_raw"])
    p.add_argument("--mc_samples",    type=int,   default=200)
    p.add_argument("--mc_draws",      type=int,   default=20)
    p.add_argument("--benchmarks",    nargs="+",  default=["mmp", "mc_raw"],
                   choices=["mmp", "mc_raw", "full_raw"])
    # bubble mode
    p.add_argument("--n_cortical",    type=int,   default=None,
                   help="Cortical bubbles; enables bubble mode (atlas default: 360)")
    p.add_argument("--n_subcortical", type=int,   default=None,
                   help="Subcortical bubbles (atlas default: 19)")
    p.add_argument("--lam_coverage",  type=float, default=0.0)
    p.add_argument("--lam_overlap",   type=float, default=0.0)
    p.add_argument("--lam_radius",    type=float, default=0.0)
    return p.parse_args()


# ── dataset ────────────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    def __init__(self, parc, raw, window_size):
        self.parc, self.raw, self.W = parc, raw, window_size

    def __len__(self):
        return len(self.parc) - self.W

    def __getitem__(self, i):
        return (torch.tensor(self.parc[i : i + self.W]),   # (W, 379)
                torch.tensor(self.raw[i  : i + self.W]),   # (W, G)
                torch.tensor(self.parc[i + self.W]),        # (379,)
                torch.tensor(self.raw[i  + self.W]))        # (G,)


# ── coordinate helpers ─────────────────────────────────────────────────────────

def build_grayordinate_coords(cifti_path):
    """
    Returns:
      coords_c: (59412, 3) fsLR sphere coords for cortical grayordinates
      coords_s: (31870, 3) MNI mm coords for subcortical grayordinates
    """
    cl, _ = hcp.mesh.sphere_left
    cr, _ = hcp.mesh.sphere_right
    coords_c = np.vstack([cl[hcp.vertex_info.grayl], cr[hcp.vertex_info.grayr]]).astype(np.float32)

    ax = nib.load(cifti_path).header.get_axis(1)
    segs = []
    for _, _, bm in ax.iter_structures():
        if bm.affine is not None:
            vox  = bm.voxel.astype(float)
            mni  = (bm.affine @ np.hstack([vox, np.ones((len(vox), 1))]).T).T[:, :3]
            segs.append(mni.astype(np.float32))
    coords_s = np.vstack(segs)  # (31870, 3)
    return coords_c, coords_s


# ── fixed-atlas loss helpers ───────────────────────────────────────────────────

def build_parcel_to_gray(labels, valid_mask):
    valid_idx  = np.where(valid_mask)[0]
    parcel_idx = labels[valid_idx] - 1
    return valid_idx, parcel_idx


def reparcellate(pred_g, valid_gray_idx, parcel_for_gray, N_parc):
    """
    Decode grayordinate predictions back to MMP parcel space via fixed atlas mean.
    pred_g: (B, 1, G)  →  (B, 1, N_parc)
    """
    B      = pred_g.shape[0]
    device = pred_g.device
    g_idx  = valid_gray_idx.to(device)
    p_idx  = parcel_for_gray.to(device)
    out    = torch.zeros(B, 1, N_parc, device=device)
    cnt    = torch.zeros(N_parc, device=device)
    cnt.scatter_add_(0, p_idx, torch.ones(len(p_idx), dtype=torch.float32, device=device))
    out.scatter_add_(2, p_idx.unsqueeze(0).unsqueeze(0).expand(B, 1, -1), pred_g[:, :, g_idx])
    return out / cnt.clamp(min=1)


def mc_raw_loss_fixed(pred, y_raw, valid_gray_idx, parcel_for_gray, k):
    sample      = torch.randint(0, len(valid_gray_idx), (k,))
    pred_at_g   = pred[:, :, parcel_for_gray[sample]]
    actual_at_g = y_raw[:, :, valid_gray_idx[sample]]
    return nn.functional.mse_loss(pred_at_g, actual_at_g)


# ── bubble loss helpers ────────────────────────────────────────────────────────

def mc_raw_loss_bubble(pred, y_raw, W_c, W_s, N_c, G_c, k):
    """
    Decode pred from bubble space to raw grayordinate space via W_c / W_s,
    sample k positions, compute MSE vs y_raw.
    pred:  (B, 1, N_c + N_s)
    y_raw: (B, 1, G_c + G_s)
    """
    G     = G_c + W_s.shape[1]
    idx   = torch.randint(0, G, (k,), device=pred.device)
    c_sel = idx[idx < G_c]
    s_sel = idx[idx >= G_c] - G_c

    parts_pred, parts_actual = [], []
    if c_sel.numel():
        parts_pred.append(pred[:, :, :N_c] @ W_c[:, c_sel])
        parts_actual.append(y_raw[:, :, c_sel])
    if s_sel.numel():
        parts_pred.append(pred[:, :, N_c:] @ W_s[:, s_sel])
        parts_actual.append(y_raw[:, :, G_c + s_sel])

    return nn.functional.mse_loss(torch.cat(parts_pred, -1),
                                  torch.cat(parts_actual, -1))


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Args:   {vars(args)}")

    use_bubbles = (args.n_cortical is not None) or (args.n_subcortical is not None)
    N_c = args.n_cortical   or 360
    N_s = args.n_subcortical or 19

    # ── atlas ──────────────────────────────────────────────────────────────────
    labels     = hcp.mmp.map_all          # (91282,) 1-indexed, 0 = medial wall
    valid_mask = labels > 0
    valid_gray_idx, parcel_for_gray = build_parcel_to_gray(labels, valid_mask)
    valid_gray_idx  = torch.tensor(valid_gray_idx,  dtype=torch.long)
    parcel_for_gray = torch.tensor(parcel_for_gray, dtype=torch.long)

    # ── data ───────────────────────────────────────────────────────────────────
    parc = np.load(args.data).astype(np.float32)   # (T, 379)
    raw  = np.load(args.raw).astype(np.float32)    # (T, G)
    T, N_parc = parc.shape
    G_c_data  = 59412                              # cortical grayordinates in CIFTI
    split = int(T * args.train_split)

    train_ds = WindowDataset(parc[:split],             raw[:split],             args.window)
    test_ds  = WindowDataset(parc[split-args.window:], raw[split-args.window:], args.window)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Train: {split} TPs → {len(train_ds)} windows  |  Test: {T-split} TPs → {len(test_ds)} windows")

    # ── model ──────────────────────────────────────────────────────────────────
    N_model = (N_c + N_s) if use_bubbles else N_parc
    model = TimeSeriesTransformer(
        input_size=N_model, dec_seq_len=1, dim_val=args.dim_val,
        n_heads=args.n_heads, n_encoder_layers=args.n_enc,
        n_decoder_layers=args.n_dec, max_seq_len=args.window,
        out_seq_len=1, num_predicted_features=N_model, batch_first=True,
    ).to(device)

    # ── bubble parcellation (optional) ─────────────────────────────────────────
    if use_bubbles:
        print(f"Bubble mode: N_c={N_c}  N_s={N_s}  — building coords...")
        coords_c, coords_s = build_grayordinate_coords(args.cifti)
        bubble_parc = BubbleParcellation(
            coords_c, coords_s, labels, N_c, N_s
        ).to(device)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(bubble_parc.parameters()), lr=args.lr
        )
        mb = args.batch_size * args.window * raw.shape[1] * 4 / 1e6
        if mb > 400:
            print(f"  Warning: x_raw per batch ≈ {mb:.0f} MB — consider --batch_size 16")
        mode_str = f"bubble(N_c={N_c}, N_s={N_s})"
    else:
        bubble_parc = None
        optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr)
        mode_str    = args.loss

    mse = nn.MSELoss()
    print(f"Training loss: {mode_str}  |  Benchmarks: {args.benchmarks}")

    # ── training loop ──────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        if bubble_parc is not None:
            bubble_parc.train()
        train_losses = []

        for x_parc, x_raw, y_parc, y_raw in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x_parc = x_parc.to(device)
            x_raw  = x_raw.to(device)
            y_parc = y_parc.unsqueeze(1).to(device)
            y_raw  = y_raw.unsqueeze(1).to(device)

            if bubble_parc is not None:
                W_c, W_s = bubble_parc.weights()
                x_in     = bubble_parc(x_raw, W_c, W_s)
                dec_in   = x_in[:, -1:, :]
                pred     = model(x_in, dec_in)

                loss = mc_raw_loss_bubble(pred, y_raw, W_c, W_s, N_c, G_c_data, args.mc_samples)
                if args.lam_coverage or args.lam_overlap or args.lam_radius:
                    R_cov, R_ov, R_rad = bubble_regularizers(W_c, W_s,
                                                               bubble_parc.log_r_c, bubble_parc.log_r_s)
                    loss = loss + args.lam_coverage * R_cov + args.lam_overlap * R_ov + args.lam_radius * R_rad
            else:
                x_in   = x_parc
                dec_in = x_in[:, -1:, :]
                pred   = model(x_in, dec_in)

                if args.loss == "mmp":
                    loss = mse(pred, y_parc)
                else:
                    loss = mc_raw_loss_fixed(pred, y_raw, valid_gray_idx, parcel_for_gray, args.mc_samples)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ── eval ───────────────────────────────────────────────────────────────
        model.eval()
        if bubble_parc is not None:
            bubble_parc.eval()
        bench_accum = {k: [] for k in args.benchmarks}

        with torch.no_grad():
            for x_parc, x_raw, y_parc, y_raw in test_dl:
                x_parc = x_parc.to(device)
                x_raw  = x_raw.to(device)
                y_parc = y_parc.unsqueeze(1).to(device)
                y_raw  = y_raw.unsqueeze(1).to(device)

                if bubble_parc is not None:
                    W_c, W_s = bubble_parc.weights()
                    x_in     = bubble_parc(x_raw, W_c, W_s)
                else:
                    W_c = W_s = None
                    x_in = x_parc

                dec_in = x_in[:, -1:, :]
                pred   = model(x_in, dec_in)

                for bm_key in args.benchmarks:
                    if bm_key == "mmp":
                        if use_bubbles:
                            pred_g   = torch.cat([pred[:, :, :N_c] @ W_c,
                                                  pred[:, :, N_c:] @ W_s], dim=-1)
                            pred_mmp = reparcellate(pred_g, valid_gray_idx, parcel_for_gray, N_parc)
                            bench_accum["mmp"].append(mse(pred_mmp, y_parc).item())
                        else:
                            bench_accum["mmp"].append(mse(pred, y_parc).item())
                    elif bm_key == "mc_raw":
                        draws = []
                        for _ in range(args.mc_draws):
                            if use_bubbles:
                                v = mc_raw_loss_bubble(pred, y_raw, W_c, W_s, N_c, G_c_data, args.mc_samples)
                            else:
                                v = mc_raw_loss_fixed(pred, y_raw, valid_gray_idx, parcel_for_gray, args.mc_samples)
                            draws.append(v.item())
                        bench_accum["mc_raw"].append(np.mean(draws))
                    elif bm_key == "full_raw":
                        if use_bubbles:
                            pred_g = torch.cat([pred[:, :, :N_c] @ W_c,
                                                pred[:, :, N_c:] @ W_s], dim=-1)
                            bench_accum["full_raw"].append(mse(pred_g, y_raw).item())
                        else:
                            pred_g   = pred[:, :, parcel_for_gray]
                            actual_g = y_raw[:, :, valid_gray_idx]
                            bench_accum["full_raw"].append(mse(pred_g, actual_g).item())

        bench_str = "  ".join(
            f"{k}: {np.mean(v):.4f}" for k, v in bench_accum.items() if v
        )
        print(f"Epoch {epoch:>2}  train {mode_str}: {np.mean(train_losses):.4f}  |  {bench_str}")

    # ── save ───────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), args.out)
    if bubble_parc is not None:
        bp_out = args.out.replace(".pt", "_bubbles.pt")
        torch.save(bubble_parc.state_dict(), bp_out)
        print(f"BubbleParcellation saved to {bp_out}")
    print(f"Done. Model saved to {args.out}")


if __name__ == "__main__":
    main()
