from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


def compute_latent_stats(h5_path, max_clips=None):
    """Per-channel mean/std over all clips and time steps -> (mean[D], std[D])."""
    s1 = s2 = None
    count = 0
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        if max_clips:
            keys = keys[:max_clips]
        for k in keys:
            x = torch.from_numpy(f[k][:]).double()  # [T, D]
            if s1 is None:
                s1 = torch.zeros(x.shape[1], dtype=torch.float64)
                s2 = torch.zeros_like(s1)
            s1 += x.sum(0)
            s2 += (x**2).sum(0)
            count += x.shape[0]
    mean = s1 / count
    var = (s2 / count - mean**2).clamp_min(1e-8)
    return mean.float(), var.sqrt().float()


class LatentSequenceDataset(Dataset):
    """Reads pre-extracted HDF5 latents [T, d]; optionally standardizes per channel."""

    def __init__(self, h5_path, sequence_length: int = 32, mean=None, std=None):
        super().__init__()
        self.h5_path = Path(h5_path)
        self.sequence_length = sequence_length
        self.mean = mean
        self.std = std

        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 cache file not found at: {self.h5_path}")

        with h5py.File(self.h5_path, "r") as f:
            self.keys = list(f.keys())

        self._h5_file = None

    def _get_h5_handle(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int) -> torch.Tensor:
        h5_file = self._get_h5_handle()
        latents = torch.from_numpy(h5_file[self.keys[idx]][:]).float()  # [T, d]

        if latents.shape[0] > self.sequence_length:
            latents = latents[: self.sequence_length]

        if self.mean is not None and self.std is not None:
            latents = (latents - self.mean) / self.std
        return latents