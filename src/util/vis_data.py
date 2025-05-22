import sys
sys.path.append("/large/naru/HOGraspNet/src")
from model.classify import TaxonomyClassifier
from dataset.dataset_mano import HOGDataset
from torch.utils.data import DataLoader
import cv2
from omegaconf import OmegaConf
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import os
from tqdm import tqdm

cfg = OmegaConf.load("src/cfg/cfg_mano.yaml")
val_dataset = HOGDataset(
    split="train", 
    db_path=cfg.dataset.db_path, 
    input_type=cfg.model.input_type, 
    mano_type=cfg.model.mano_type, 
    preload=cfg.dataset.preload, 
    noise_level=cfg.train.noise_level
)
val_dataloader = DataLoader(
    val_dataset, 
    batch_size=cfg.train.batch_size, 
    shuffle=False, 
    num_workers=cfg.dataset.num_workers, 
    pin_memory=False
)

all_mano = []

for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
    mano = batch["mano_pose"]  # shape (B, 45)
    mano_np = mano.cpu().numpy()  # (B, 45)
    all_mano.append(mano_np)

all_mano = np.concatenate(all_mano, axis=0)  # shape (N_samples, 45)

# 各次元ごとのmeanとstd
mean = np.mean(all_mano, axis=0)  # shape (45,)
std = np.std(all_mano, axis=0)    # shape (45,)

print(f"Overall mean (45-dim): {mean}")
print(f"Overall std  (45-dim): {std}")
os.makedirs("mano_mean_std", exist_ok=True)
# 保存したい場合
np.save("mano_mean_std/mano_mean.npy", mean)
np.save("mano_mean_std/mano_std.npy", std)
print("Saved mean and std to mano_mean.npy and mano_std.npy")