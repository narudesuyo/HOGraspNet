import os
import torch
from torch.utils.data import Dataset
import h5py
import sys
sys.path.append("/large/naru/HOGraspNet")
from src.dataset.utils import normalize_vertices, rotate_vertices_xyz, move_root_to_origin, add_noise

class HOGDataset(Dataset):
    def __init__(self, split, db_path, contact_bin=False, contact=True, vertices_aug=True):
        self.h5_path = f'{db_path}/{split}.h5'
        self.h5_file = None
        self.contact_bin = contact_bin
        self.contact = contact
        self.vertices_aug = vertices_aug
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
        if self.vertices_aug:
            vertices = normalize_vertices(vertices)
            vertices = rotate_vertices_xyz(vertices)
        if self.contact_bin:
            contact = torch.clamp((torch.from_numpy(self.h5_file[k]["contact"][:]) * 10).long(), max=9).unsqueeze(-1)
        else:
            contact = torch.from_numpy(self.h5_file[k]["contact"][:]).float().unsqueeze(-1)
        one_hot = torch.zeros(33)
        one_hot[class_id - 1] = 1.0
        combined_data = torch.cat([vertices, contact], dim=-1)

        if self.contact:
            return combined_data, one_hot
        else:
            return vertices, one_hot

    # def __del__(self):
    #     self.h5_file.close()


class HOGDatasetAll(Dataset):
    def __init__(self, split, db_path, contact_bin=False, contact=False, vertices_aug=False, max_angle=30, noise_level=0):
        self.h5_path = f'{db_path}/{split}.h5'
        self.contact_bin = contact_bin
        self.contact = contact
        self.vertices_aug = vertices_aug
        self.max_angle = max_angle
        self.noise_level = noise_level

        # あらかじめ key だけ一覧取得
        with h5py.File(self.h5_path, 'r') as f:
            self.length = f['mano_hamer'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as h5_file:
            vertices = torch.from_numpy(h5_file["mano_hamer"][idx]).float()
            class_id = int(h5_file["taxonomy_id"][idx])

            # if self.vertices_aug:
            #     vertices = normalize_vertices(vertices)
            #     vertices = rotate_vertices_xyz(vertices)
            vertices = move_root_to_origin(vertices)

            if self.vertices_aug:
                vertices = rotate_vertices_xyz(vertices, max_angle=self.max_angle)
            if self.noise_level > 0:
                vertices = add_noise(vertices, noise_level=self.noise_level)

            # if self.contact_bin:
            #     contact = torch.clamp(
            #         (torch.from_numpy(h5_file["contact"][idx]) * 10).long(), max=9
            #     ).unsqueeze(-1)
            # else:
            #     contact = torch.from_numpy(h5_file["contact"][idx]).float().unsqueeze(-1)

        one_hot = torch.zeros(33)
        one_hot[class_id - 1] = 1.0
        # combined_data = torch.cat([vertices, contact], dim=-1)

        # if self.contact:
        #     return combined_data, one_hot
        # else:
        # print(f"vertices: {vertices}")
        # print(f"one_hot: {one_hot}")
        return vertices, class_id