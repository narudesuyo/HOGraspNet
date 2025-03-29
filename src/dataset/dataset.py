import os
import torch
from torch.utils.data import Dataset
import h5py

class HOGDataset(Dataset):
    def __init__(self, split, db_path, contact_bin=False):
        self.h5_path = f'{db_path}/processed_{split}.h5'
        self.h5_file = None
        self.contact_bin = contact_bin
        with h5py.File(self.h5_path, 'r') as f:
            self.keys = list(f.keys())

    def __len__(self):
        return len(self.keys)  # keysの長さを返す

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')  # ← lazy open（worker内で実行）
        k = self.keys[idx]
        vertices = torch.from_numpy(self.h5_file[k]["vertices"][:]).float()
        class_id = int(self.h5_file[k]["class_id"][()])
        if self.contact_bin:
            contact = torch.clamp((torch.from_numpy(self.h5_file[k]["contact"][:]) * 10).long(), max=9).unsqueeze(-1)
        else:
            contact = torch.from_numpy(self.h5_file[k]["contact"][:]).float().unsqueeze(-1)
        one_hot = torch.zeros(33)
        one_hot[class_id - 1] = 1.0
        combined_data = torch.cat([vertices, contact], dim=-1)

        return combined_data, one_hot

    # def __del__(self):
    #     self.h5_file.close()
