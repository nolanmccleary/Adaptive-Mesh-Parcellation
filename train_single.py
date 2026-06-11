"""
Single-session sanity check: train on first 90% of one scan, test on last 10%.
Uses the same model architecture as the paper, float32 for MPS speed.
"""

import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import hcp_utils as hcp
from model import TimeSeriesTransformer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",        default="data/100307_REST1_LR_p.npy")
    p.add_argument("--raw",         default="data/100307_REST1_LR_raw.npy",
                   help="Raw grayordinate file for MC loss and benchmarks")
    p.add_argument("--window",      type=int,   default=30)
    p.add_argument("--train_split", type=float, default=0.9)
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--dim_val",     type=int,   default=760)
    p.add_argument("--n_heads",     type=int,   default=8)
    p.add_argument("--n_enc",       type=int,   default=4)
    p.add_argument("--n_dec",       type=int,   default=4)
    p.add_argument("--out",         default="data/single_session_model.pt")
    p.add_argument("--loss",        default="mc_raw",
                   choices=["mmp", "mc_raw"],
                   help="Training loss: mmp=full parcel MSE, mc_raw=MC over raw grayordinates")
    p.add_argument("--mc_samples",  type=int,   default=200,
                   help="Grayordinates sampled per MC draw (default 200)")
    p.add_argument("--mc_draws",    type=int,   default=20,
                   help="MC draws per batch at eval time (default 20)")
    p.add_argument("--benchmarks",  nargs="+",  default=["mmp", "mc_raw"],
                   choices=["mmp", "mc_raw", "full_raw"],
                   help="Loss benchmarks reported at end of each epoch")
    return p.parse_args()


class WindowDataset(Dataset):
    def __init__(self, parc, raw, window_size):
        self.parc = parc   # (T, 379)
        self.raw  = raw    # (T, 91282)
        self.W    = window_size

    def __len__(self):
        return len(self.parc) - self.W

    def __getitem__(self, i):
        x_parc = self.parc[i : i + self.W]   # (W, 379)  — model input
        y_parc = self.parc[i + self.W]        # (379,)    — parcel target
        y_raw  = self.raw[i + self.W]         # (G,)      — grayordinate target
        return torch.tensor(x_parc), torch.tensor(y_parc), torch.tensor(y_raw)


def build_parcel_to_gray(labels, valid_mask):
    """
    Returns parcel_to_gray: (G_valid,) int array mapping valid grayordinate
    index → parcel index (0-based) in model output.
    """
    valid_idx = np.where(valid_mask)[0]           # grayordinate indices (0-based)
    parcel_idx = labels[valid_idx] - 1            # 0-based parcel indices
    return valid_idx, parcel_idx


def mc_raw_loss(pred, y_raw, valid_gray_idx, parcel_for_gray, k, device):
    """
    Sample k random valid grayordinates. For each, look up its parcel prediction.
    MSE vs actual raw grayordinate value.
    pred:           (B, 1, N)
    y_raw:          (B, 1, G)
    valid_gray_idx: (G_valid,) — indices into dim G
    parcel_for_gray:(G_valid,) — corresponding parcel indices into dim N
    """
    sample = torch.randint(0, len(valid_gray_idx), (k,))
    g_idx = valid_gray_idx[sample]         # grayordinate positions
    p_idx = parcel_for_gray[sample]        # corresponding parcel positions

    pred_at_g  = pred[:, :, p_idx]        # (B, 1, k)
    actual_at_g = y_raw[:, :, g_idx]      # (B, 1, k)
    return nn.functional.mse_loss(pred_at_g, actual_at_g)


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Args:   {vars(args)}")

    # ── atlas ──────────────────────────────────────────────────────────────────
    labels     = hcp.mmp.map_all                          # (91282,) 1-indexed, 0=medial wall
    valid_mask = labels > 0                               # exclude medial wall
    valid_gray_idx, parcel_for_gray = build_parcel_to_gray(labels, valid_mask)
    valid_gray_idx  = torch.tensor(valid_gray_idx,  dtype=torch.long)
    parcel_for_gray = torch.tensor(parcel_for_gray, dtype=torch.long)
    print(f"Valid grayordinates (non-medial-wall): {valid_mask.sum()} / {len(labels)}")

    # ── data ───────────────────────────────────────────────────────────────────
    parc = np.load(args.data).astype(np.float32)   # (1200, 379)
    raw  = np.load(args.raw).astype(np.float32)    # (1200, 91282)
    T, N = parc.shape
    split = int(T * args.train_split)

    train_ds = WindowDataset(parc[:split],            raw[:split],            args.window)
    test_ds  = WindowDataset(parc[split-args.window:], raw[split-args.window:], args.window)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)
    print(f"Train: {split} timepoints → {len(train_ds)} windows")
    print(f"Test:  {T-split} timepoints → {len(test_ds)} windows")

    # ── model ──────────────────────────────────────────────────────────────────
    model = TimeSeriesTransformer(
        input_size=N, dec_seq_len=1, dim_val=args.dim_val,
        n_heads=args.n_heads, n_encoder_layers=args.n_enc,
        n_decoder_layers=args.n_dec, max_seq_len=args.window,
        out_seq_len=1, num_predicted_features=N, batch_first=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mmp_loss  = nn.MSELoss()
    print(f"Training loss: {args.loss}  |  Benchmarks: {args.benchmarks}")

    # ── helpers ────────────────────────────────────────────────────────────────
    def train_loss_fn(pred, y_parc, y_raw):
        if args.loss == "mmp":
            return mmp_loss(pred, y_parc)
        else:  # mc_raw
            return mc_raw_loss(pred, y_raw, valid_gray_idx, parcel_for_gray,
                               args.mc_samples, device)

    def run_benchmarks(pred, y_parc, y_raw):
        results = {}
        if "mmp" in args.benchmarks:
            results["mmp"] = mmp_loss(pred, y_parc).item()
        if "mc_raw" in args.benchmarks:
            results["mc_raw"] = np.mean([
                mc_raw_loss(pred, y_raw, valid_gray_idx, parcel_for_gray,
                            args.mc_samples, device).item()
                for _ in range(args.mc_draws)
            ])
        if "full_raw" in args.benchmarks:
            pred_g  = pred[:, :, parcel_for_gray]
            actual_g = y_raw[:, :, valid_gray_idx]
            results["full_raw"] = mmp_loss(pred_g, actual_g).item()
        return results

    # ── training loop ──────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y_parc, y_raw in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x      = x.to(device)
            y_parc = y_parc.unsqueeze(1).to(device)
            y_raw  = y_raw.unsqueeze(1).to(device)
            dec_in = x[:, -1:, :]
            pred   = model(x, dec_in)
            loss   = train_loss_fn(pred, y_parc, y_raw)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        bench_accum = {k: [] for k in args.benchmarks}
        with torch.no_grad():
            for x, y_parc, y_raw in test_dl:
                x      = x.to(device)
                y_parc = y_parc.unsqueeze(1).to(device)
                y_raw  = y_raw.unsqueeze(1).to(device)
                dec_in = x[:, -1:, :]
                pred   = model(x, dec_in)
                b = run_benchmarks(pred, y_parc, y_raw)
                for k, v in b.items():
                    bench_accum[k].append(v)

        bench_str = "  ".join(f"{k}: {np.mean(v):.4f}" for k, v in bench_accum.items())
        print(f"Epoch {epoch:>2}  train {args.loss}: {np.mean(train_losses):.4f}  |  {bench_str}")

    torch.save(model.state_dict(), args.out)
    print(f"\nDone. Model saved to {args.out}")


if __name__ == "__main__":
    main()
