import h5py
import torch
from torch.utils.data import Dataset
import torchvision
import sys
import numpy as np
sys.path.append("/large/naru/HOGraspNet/src")
from dataset.utils import rotation_matrix_to_axis_angle, add_noise
class HOGDataset(Dataset):
    def __init__(self, split, db_path, transform=None, input_type=["mano", "image"], mano_type="gt", preload=False, noise_level=0,
                 mano_mean="/large/naru/HOGraspNet/mano_mean_std/mano_mean.npy",
                 mano_std="/large/naru/HOGraspNet/mano_mean_std/mano_std.npy"):
        self.split = split
        self.h5_path = f'{db_path}/{split}.h5'
        self.transform = transform
        self.input_type = input_type
        self.mano_type = mano_type
        self.preload = preload
        self.noise_level = noise_level
        self.mano_mean = np.load(mano_mean)
        self.mano_std = np.load(mano_std)
        with h5py.File(self.h5_path, 'r') as f:
            self.length = f['taxonomy_id'].shape[0]
            if preload:
                # 全データをメモリに読み込む
                self.taxonomy_id = f['taxonomy_id'][:]
                if "image" in input_type:
                    self.rgb = f['rgb'][:]
                if "mano" in input_type:
                    if mano_type == "gt":
                        self.mano_pose = f['mano_pose_gt'][:]
                    elif mano_type == "pred":
                        self.mano_pose = f['mano_pose_pred'][:]
                    else:
                        raise ValueError(f"Invalid mano_type: {mano_type}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.preload:
            class_id = int(self.taxonomy_id[idx])
            rgb = torch.from_numpy(self.rgb[idx]).permute(2, 0, 1) if "image" in self.input_type else torch.zeros(3,224,224)
            if self.transform and rgb is not None:
                rgb = self.transform(rgb)
            mano_pose = self.mano_pose[idx] if "mano" in self.input_type else torch.zeros(1,45)
        
        else:
            with h5py.File(self.h5_path, 'r') as f:
                class_id = int(f['taxonomy_id'][idx])
                if "image" in self.input_type:
                    rgb = torch.from_numpy(f['rgb'][idx]).permute(2, 0, 1)
                    if self.transform:
                        rgb = self.transform(rgb)
                else:
                    rgb = torch.zeros(3,224,224)
                if "mano" in self.input_type:
                    if self.mano_type == "gt":
                        mano_pose = f['mano_pose_gt'][idx]
                        if self.noise_level > 0 and self.split != "test":
                            mano_pose = add_noise(mano_pose, self.noise_level)
                        if self.mano_mean is not None and self.mano_std is not None:
                            mano_pose = (mano_pose - self.mano_mean) / self.mano_std
                        # mano_pose = torch.tensor(mano_pose, dtype=torch.float32)
                    elif self.mano_type == "pred":
                        mano_pose = rotation_matrix_to_axis_angle(f['mano_pose_pred'][idx]).float()
                        if self.noise_level > 0 and self.split != "test":
                            mano_pose = add_noise(mano_pose, self.noise_level)
                        if self.mano_mean is not None and self.mano_std is not None:
                            mano_pose = (mano_pose - self.mano_mean) / self.mano_std
                        # mano_pose = torch.tensor(mano_pose, dtype=torch.float32)
                    else:
                        raise ValueError(f"Invalid mano_type: {self.mano_type}")
                else:
                    mano_pose = torch.zeros(1,45)
            # if self.noise_level > 0 and self.split != "test":
            #     mano_pose = add_noise(mano_pose, self.noise_level)
        # print(f"mano_pose: {mano_pose.shape}")
        if isinstance(mano_pose, np.ndarray):
            mano_pose = torch.from_numpy(mano_pose)
        mano_pose = mano_pose.float()
        # print(f"img_type: {rgb.dtype}")
        # print(f"mano_pose_type: {mano_pose.dtype}")
        batch = {
            "img_feat": rgb,
            "mano_pose": mano_pose,
            "taxonomy_id": class_id,
        }
        return batch