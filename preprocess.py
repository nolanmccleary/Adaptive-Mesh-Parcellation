"""
Preprocessing pipeline for HCP resting-state fMRI CIFTI data.

Steps:
    1. Load CIFTI dense timeseries (.dtseries.nii)
    2. Bandpass filter (0.01–0.1 Hz)
    3. Z-score normalization (per grayordinate, over time)
    4. Save raw grayordinates → _raw.npy
    5. Parcellate to 379 regions (HCP-MMP 360 cortical + 19 subcortical)
    6. Z-score (per parcel)
    7. Save parcellated → _p.npy
    8. Save subcortical MNI coords → subcortical_coords.npy (once per cache dir)
"""

import sys
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import signal

TR = 0.72
LOWCUT = 0.01
HIGHCUT = 0.1


def bandpass_filter(data, lowcut, highcut, tr):
    fs = 1.0 / tr
    nyq = fs / 2.0
    b, a = signal.butter(4, [lowcut / nyq, highcut / nyq], btype='band')
    print(f"  Bandpass: {lowcut}–{highcut} Hz  (fs={fs:.3f} Hz, order=4)")
    return signal.filtfilt(b, a, data, axis=0).astype(np.float32)


def zscore(data):
    mu = data.mean(axis=0, keepdims=True)
    sd = data.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    result = ((data - mu) / sd).astype(np.float32)
    print(f"  Z-score: mean={result.mean():.4f}  std={result.std():.4f}")
    return result


def parcellate(data):
    import hcp_utils as hcp
    labels    = hcp.mmp.map_all
    n_parcels = int(labels.max())
    T         = data.shape[0]
    out       = np.zeros((T, n_parcels), dtype=np.float32)
    for p in range(1, n_parcels + 1):
        mask = labels == p
        if mask.sum() > 0:
            out[:, p - 1] = data[:, mask].mean(axis=1)
    print(f"  Parcellated: {data.shape} → {out.shape}")
    return out


def save_subcortical_coords(cifti_img, out_path):
    """Extract MNI coords for subcortical grayordinates. Identical across HCP subjects."""
    out_path = Path(out_path)
    if out_path.exists():
        return
    ax   = cifti_img.header.get_axis(1)
    segs = []
    for _, _, bm in ax.iter_structures():
        if bm.affine is not None:
            vox = bm.voxel.astype(float)
            mni = (bm.affine @ np.hstack([vox, np.ones((len(vox), 1))]).T).T[:, :3]
            segs.append(mni.astype(np.float32))
    coords = np.vstack(segs)
    np.save(str(out_path), coords)
    print(f"  Subcortical coords: {out_path.name}  shape={coords.shape}")


def run(cifti_path, out_path):
    raw_path    = out_path.replace('_p.npy', '_raw.npy')
    coords_path = Path(out_path).parent / 'subcortical_coords.npy'

    print(f"\n[1] Loading {cifti_path}")
    img  = nib.load(cifti_path)
    data = img.get_fdata(dtype=np.float32)
    print(f"  Loaded: {data.shape}  dtype={data.dtype}")

    print("\n[2] Bandpass filter")
    data = bandpass_filter(data, LOWCUT, HIGHCUT, TR)

    print("\n[3] Z-score (grayordinate level)")
    data = zscore(data)

    print(f"\n[4] Save raw grayordinates → {raw_path}")
    np.save(raw_path, data)
    print(f"  Saved: {data.shape}  ({data.nbytes / 1e6:.0f} MB)")

    print("\n[5] Parcellate")
    parcellated = parcellate(data)

    print("\n[6] Z-score (parcel level)")
    parcellated = zscore(parcellated)

    print(f"\n[7] Save parcellated → {out_path}")
    np.save(out_path, parcellated)
    print(f"  Saved: {parcellated.shape}")

    print("\n[8] Save subcortical coords")
    save_subcortical_coords(img, coords_path)

    return parcellated


if __name__ == "__main__":
    cifti = sys.argv[1] if len(sys.argv) > 1 else "data/100307_REST1_LR.dtseries.nii"
    out   = sys.argv[2] if len(sys.argv) > 2 else "data/100307_REST1_LR_p.npy"
    result = run(cifti, out)

    print("\n--- Sanity checks (parcellated) ---")
    print(f"Shape:   {result.shape}  (expect (1200, 379))")
    print(f"Mean:    {result.mean():.4f}  (expect ~0)")
    print(f"Std:     {result.std():.4f}  (expect ~1)")
    print(f"NaNs:    {np.isnan(result).sum()}")
    print(f"Min/Max: {result.min():.3f} / {result.max():.3f}")
