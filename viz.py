"""
Post-process per-epoch bubble artifact npz files into MP4 animations.

Usage:
  python viz.py data/<run>/artifacts/ [--mode 2d|3d|both] [--fps 2] [--decimate 1]
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import hcp_utils as hcp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("artifacts_dir", help="run_dir/artifacts/")
    p.add_argument("--mode",     default="both", choices=["2d", "3d", "both"])
    p.add_argument("--fps",      type=int, default=2)
    p.add_argument("--decimate", type=int, default=1)
    return p.parse_args()


def _load_coords():
    grayl = hcp.vertex_info.grayl
    grayr = hcp.vertex_info.grayr

    flat_l, _ = hcp.mesh.flat_left
    flat_r, _ = hcp.mesh.flat_right
    yz_l = flat_l[grayl][:, 1:]
    yz_r = flat_r[grayr][:, 1:]

    r_offset = float(yz_l[:, 0].max() - yz_r[:, 0].min()) + 20.0
    yz_r_shifted = yz_r.copy()
    yz_r_shifted[:, 0] += r_offset
    yz_combined = np.vstack([yz_l, yz_r_shifted])  # (G_c, 2)

    sphere_l, _ = hcp.mesh.sphere_left
    sphere_r, _ = hcp.mesh.sphere_right
    sphere_c = np.vstack([sphere_l[grayl], sphere_r[grayr]])  # (G_c, 3)

    return yz_combined, sphere_c


def _bubble_field(centers, log_r, coords):
    """Max unnormalized Gaussian over bubbles at each coord. Peaks fixed at 1."""
    r2 = np.exp(log_r) ** 2
    a2 = (centers ** 2).sum(1)
    b2 = (coords  ** 2).sum(1)
    ab = centers @ coords.T
    d2 = (a2[:, None] + b2[None, :] - 2 * ab).clip(min=0)
    return np.exp(-d2 / (2 * r2[:, None])).max(0)


def _style_3d(ax, title, elev, azim):
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('0.75')
    ax.yaxis.pane.set_edgecolor('0.75')
    ax.zaxis.pane.set_edgecolor('0.75')
    ax.set_title(title, fontsize=8, pad=2)


def _render_2d(npz_path, yz_combined, sphere_c, subcoords, dec, out_png):
    """Flat-sheet view: predicted spikes (z=field), true heatmap (z=0)."""
    d         = np.load(npz_path)
    pred      = d['pred_intensities']
    true      = d['true_intensities']
    centers_c = d['centers_c']
    log_r_c   = d['log_r_c']
    centers_s = d['centers_s']
    log_r_s   = d['log_r_s']

    G_c = len(sphere_c)
    pred_c, pred_s = pred[:G_c], pred[G_c:]
    true_c, true_s = true[:G_c], true[G_c:]

    field_c = _bubble_field(centers_c, log_r_c, sphere_c)
    field_s = _bubble_field(centers_s, log_r_s, subcoords)

    sub_xy  = subcoords[:, :2]
    vmin_c  = float(np.percentile(np.concatenate([pred_c, true_c]), 1))
    vmax_c  = float(np.percentile(np.concatenate([pred_c, true_c]), 99))
    vmin_s  = float(np.percentile(np.concatenate([pred_s, true_s]), 1))
    vmax_s  = float(np.percentile(np.concatenate([pred_s, true_s]), 99))
    epoch   = int(npz_path.stem.split('_')[-1])

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.08, wspace=0.05,
                            left=0.02, right=0.98, top=0.94, bottom=0.02)
    fig.suptitle(f"Epoch {epoch:03d}  [2D]", fontsize=10, y=0.98)

    # predicted — spike field
    ax = fig.add_subplot(gs[0, 0], projection='3d')
    ax.scatter(yz_combined[::dec, 0], yz_combined[::dec, 1], field_c[::dec],
               c=field_c[::dec], cmap='hot', s=0.3, vmin=0, vmax=1,
               rasterized=True, linewidths=0, depthshade=False)
    _style_3d(ax, "Predicted  cortex (L+R)", elev=35, azim=-60)

    ax = fig.add_subplot(gs[0, 1], projection='3d')
    ax.scatter(sub_xy[::dec, 0], sub_xy[::dec, 1], field_s[::dec],
               c=field_s[::dec], cmap='hot', s=1.5, vmin=0, vmax=1,
               rasterized=True, linewidths=0, depthshade=False)
    _style_3d(ax, "Predicted  subcortical", elev=35, azim=-60)

    # true — flat heatmap
    ax = fig.add_subplot(gs[1, 0], projection='3d')
    ax.scatter(yz_combined[::dec, 0], yz_combined[::dec, 1],
               np.zeros(len(yz_combined[::dec])),
               c=true_c[::dec], cmap='RdBu_r', s=0.3,
               vmin=vmin_c, vmax=vmax_c, rasterized=True, linewidths=0, depthshade=False)
    _style_3d(ax, "True  cortex (L+R)", elev=35, azim=-60)

    ax = fig.add_subplot(gs[1, 1], projection='3d')
    ax.scatter(sub_xy[::dec, 0], sub_xy[::dec, 1],
               np.zeros(len(sub_xy[::dec])),
               c=true_s[::dec], cmap='RdBu_r', s=1.5,
               vmin=vmin_s, vmax=vmax_s, rasterized=True, linewidths=0, depthshade=False)
    _style_3d(ax, "True  subcortical", elev=35, azim=-60)

    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)


def _render_3d(npz_path, sphere_c, subcoords, dec, out_png):
    """True 3D view: sphere surface for cortex, MNI space for subcortical."""
    d         = np.load(npz_path)
    pred      = d['pred_intensities']
    true      = d['true_intensities']
    centers_c = d['centers_c']
    log_r_c   = d['log_r_c']
    centers_s = d['centers_s']
    log_r_s   = d['log_r_s']

    G_c = len(sphere_c)
    pred_c, pred_s = pred[:G_c], pred[G_c:]
    true_c, true_s = true[:G_c], true[G_c:]

    field_c = _bubble_field(centers_c, log_r_c, sphere_c)
    field_s = _bubble_field(centers_s, log_r_s, subcoords)

    vmin_c = float(np.percentile(np.concatenate([pred_c, true_c]), 1))
    vmax_c = float(np.percentile(np.concatenate([pred_c, true_c]), 99))
    vmin_s = float(np.percentile(np.concatenate([pred_s, true_s]), 1))
    vmax_s = float(np.percentile(np.concatenate([pred_s, true_s]), 99))
    epoch  = int(npz_path.stem.split('_')[-1])

    sc = sphere_c[::dec]
    su = subcoords[::dec]

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.08, wspace=0.05,
                            left=0.02, right=0.98, top=0.94, bottom=0.02)
    fig.suptitle(f"Epoch {epoch:03d}  [3D]", fontsize=10, y=0.98)

    # predicted — field strength colored on surface
    ax = fig.add_subplot(gs[0, 0], projection='3d')
    ax.scatter(sc[:, 0], sc[:, 1], sc[:, 2],
               c=field_c[::dec], cmap='hot', s=0.3, vmin=0, vmax=1,
               rasterized=True, linewidths=0, depthshade=True)
    _style_3d(ax, "Predicted  cortex (L+R)", elev=20, azim=-60)

    ax = fig.add_subplot(gs[0, 1], projection='3d')
    ax.scatter(su[:, 0], su[:, 1], su[:, 2],
               c=field_s[::dec], cmap='hot', s=3, vmin=0, vmax=1,
               rasterized=True, linewidths=0, depthshade=True)
    _style_3d(ax, "Predicted  subcortical", elev=20, azim=-60)

    # true — intensity colored on surface
    ax = fig.add_subplot(gs[1, 0], projection='3d')
    ax.scatter(sc[:, 0], sc[:, 1], sc[:, 2],
               c=true_c[::dec], cmap='RdBu_r', s=0.3,
               vmin=vmin_c, vmax=vmax_c, rasterized=True, linewidths=0, depthshade=True)
    _style_3d(ax, "True  cortex (L+R)", elev=20, azim=-60)

    ax = fig.add_subplot(gs[1, 1], projection='3d')
    ax.scatter(su[:, 0], su[:, 1], su[:, 2],
               c=true_s[::dec], cmap='RdBu_r', s=3,
               vmin=vmin_s, vmax=vmax_s, rasterized=True, linewidths=0, depthshade=True)
    _style_3d(ax, "True  subcortical", elev=20, azim=-60)

    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)


def _encode(frames_dir, output, fps):
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', str(frames_dir / 'frame_%04d.png'),
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        str(output),
    ]
    print(f"\n{' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Saved: {output}")


def main():
    args          = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    per_epoch_dir = artifacts_dir / 'per_epoch'

    npzs = sorted(per_epoch_dir.glob('epoch_*.npz'))
    if not npzs:
        print(f"No epoch_*.npz found in {per_epoch_dir}")
        return

    subcoords_path = artifacts_dir / 'subcortical_coords.npy'
    if not subcoords_path.exists():
        raise FileNotFoundError(f"{subcoords_path} not found")
    subcoords = np.load(str(subcoords_path))

    print("Loading coords...")
    yz_combined, sphere_c = _load_coords()

    do_2d = args.mode in ("2d", "both")
    do_3d = args.mode in ("3d", "both")

    if do_2d:
        frames_2d = artifacts_dir / 'frames_2d'
        frames_2d.mkdir(exist_ok=True)
        print(f"\nRendering {len(npzs)} 2D frame(s)...")
        for i, npz in enumerate(npzs):
            out_png = frames_2d / f'frame_{i:04d}.png'
            print(f"  {npz.name} → {out_png.name}")
            _render_2d(npz, yz_combined, sphere_c, subcoords, args.decimate, out_png)
        _encode(frames_2d, artifacts_dir / 'evolution_2d.mp4', args.fps)

    if do_3d:
        frames_3d = artifacts_dir / 'frames_3d'
        frames_3d.mkdir(exist_ok=True)
        print(f"\nRendering {len(npzs)} 3D frame(s)...")
        for i, npz in enumerate(npzs):
            out_png = frames_3d / f'frame_{i:04d}.png'
            print(f"  {npz.name} → {out_png.name}")
            _render_3d(npz, sphere_c, subcoords, args.decimate, out_png)
        _encode(frames_3d, artifacts_dir / 'evolution_3d.mp4', args.fps)


if __name__ == '__main__':
    main()
