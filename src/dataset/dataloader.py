"""HOGraspNet dataset dataloader."""

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import torch
from torch.utils.data import DataLoader
import numpy as np

class HOGDataLoader:
    def __init__(self, split, db_path, batch_size=64, num_workers=4, shuffle=True, contact_bin=False, contact=True, vertices_aug=False, max_angle=0, noise_level=0):
        """Constructor.
        Args:
            setup: Setup name. 'travel_all', 's0', 's1', 's2', 's3', or 's4'
            split: Split name. 'train', 'val', or 'test'
            db_path: path to dataset folder.
            use_aug: Use crop&augmented rgb data if exists.
            load_pkl: Use saved pkl if exists.
        """
        from dataset.dataset import HOGDatasetAll
        self.dataset = HOGDatasetAll(split, db_path, contact_bin, contact, vertices_aug, max_angle, noise_level)
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers, 
            pin_memory=True
        )

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return iter(self.dataloader)

def get_dataloader(setup, split, db_path, batch_size=32, num_workers=4, shuffle=True, use_aug=False, load_pkl=True, path_pkl=None):
    """Get dataloader.
    Args:
        setup: Setup name. 'travel_all', 's0', 's1', 's2', 's3', or 's4'
        split: Split name. 'train', 'val', or 'test'
        db_path: path to dataset folder.
        batch_size: batch size.
        num_workers: number of workers.
        shuffle: shuffle data.
        use_aug: Use crop&augmented rgb data if exists.
        load_pkl: Use saved pkl if exists.
    Returns:
        dataloader
    """
    dataset = HOGDataLoader(setup, split, db_path, use_aug, load_pkl, path_pkl)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

if __name__ == '__main__':
    # Test dataloader
    setup = 's2'
    split = 'test'
    db_path = "/large/naru/HOGraspNet/data"
    
    dataloader = get_dataloader(setup, split, db_path, batch_size=2)
    for data, labels in dataloader:
        print("Combined data shape:", data.shape)  # Expected: (batch_size, 778, 4)
        print("Labels shape:", labels.shape)  # Expected: (batch_size, 33)
        break
