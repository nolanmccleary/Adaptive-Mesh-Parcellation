"""
Single-session sanity check: train on first 90% of one scan, test on last 10%.
Uses the same model architecture as the paper, float32 for CPU speed.
"""

import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import TimeSeriesTransformer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",        default="data/100307_REST1_LR_p.npy")
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
    p.add_argument("--mc_samples",  type=int, default=None,
                   help="If set, use Monte Carlo MSE loss sampling this many regions per step")
    p.add_argument("--mc_draws",    type=int, default=50,
                   help="Number of MC draws per batch at eval time (default 50)")
    return p.parse_args()


class WindowDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.W    = window_size

    def __len__(self):
        return len(self.data) - self.W

    def __getitem__(self, i):
        x = self.data[i : i + self.W]
        y = self.data[i + self.W]
        return torch.tensor(x), torch.tensor(y)


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Args:   {vars(args)}")

    data  = np.load(args.data).astype(np.float32)
    T     = data.shape[0]
    N     = data.shape[1]
    split = int(T * args.train_split)

    train_data = data[:split]
    test_data  = data[split - args.window:]

    print(f"Train timepoints: {split}  →  {len(train_data) - args.window} windows")
    print(f"Test  timepoints: {T - split}  →  {len(test_data) - args.window} windows")

    train_ds = WindowDataset(train_data, args.window)
    test_ds  = WindowDataset(test_data,  args.window)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    model = TimeSeriesTransformer(
        input_size=N,
        dec_seq_len=1,
        dim_val=args.dim_val,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_enc,
        n_decoder_layers=args.n_dec,
        max_seq_len=args.window,
        out_seq_len=1,
        num_predicted_features=N,
        batch_first=True,
    ).to(device)

    full_mse  = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_mc    = args.mc_samples is not None
    mode_str  = f"MC-MSE (k={args.mc_samples})" if use_mc else "full MSE"
    print(f"Loss: {mode_str}")

    def mc_loss(pred, target, k):
        idx = torch.randint(0, pred.shape[-1], (k,), device=pred.device)
        return full_mse(pred[..., idx], target[..., idx])

    def eval_both(pred, target):
        mse = full_mse(pred, target).item()
        mc  = np.mean([mc_loss(pred, target, args.mc_samples).item()
                       for _ in range(args.mc_draws)]) if use_mc else mse
        return mse, mc

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs} train", leave=False):
            x      = x.to(device)
            y      = y.unsqueeze(1).to(device)
            dec_in = x[:, -1:, :]
            pred   = model(x, dec_in)
            loss   = mc_loss(pred, y, args.mc_samples) if use_mc else full_mse(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        test_mse_list, test_mc_list = [], []
        with torch.no_grad():
            for x, y in test_dl:
                x      = x.to(device)
                y      = y.unsqueeze(1).to(device)
                dec_in = x[:, -1:, :]
                pred   = model(x, dec_in)
                mse, mc = eval_both(pred, y)
                test_mse_list.append(mse)
                test_mc_list.append(mc)

        if use_mc:
            print(f"Epoch {epoch:>2}  train MC-MSE: {np.mean(train_losses):.4f}  "
                  f"test full-MSE: {np.mean(test_mse_list):.4f}  "
                  f"test MC-MSE: {np.mean(test_mc_list):.4f}")
        else:
            print(f"Epoch {epoch:>2}  train MSE: {np.mean(train_losses):.4f}  "
                  f"test MSE: {np.mean(test_mse_list):.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"\nDone. Model saved to {args.out}")


if __name__ == "__main__":
    main()
