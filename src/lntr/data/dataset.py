from pathlib import Path
import h5py
import torch
from torch.utils.data import Dataset

class LatentSequenceDataset(Dataset):
    """PyTorch Dataset reading pre-extracted HDF5 latent tensors (T=32, d=1024)."""
    def __init__(self, h5_path: str | Path, sequence_length: int = 32):
        super().__init__()
        self.h5_path = Path(h5_path)
        self.sequence_length = sequence_length

        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 cache file not found at: {self.h5_path}")

        # Index all keys without locking file across worker threads
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
        clip_id = self.keys[idx]
        
        latents_np = h5_file[clip_id][:]  # [32, 1024]
        latents = torch.from_numpy(latents_np).float()

        if latents.shape[0] > self.sequence_length:
            latents = latents[:self.sequence_length]

        return latents