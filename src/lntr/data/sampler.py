import torch
from torch.utils.data import Sampler
import random

class TrajectoryWindowSampler(Sampler):
    """Custom sampler for sliding-window sequence batching."""
    def __init__(self, data_source, batch_size: int, shuffle: bool = True, seed: int = 42):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            yield indices[i:i + self.batch_size]

    def __len__(self):
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size