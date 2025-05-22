import sys
sys.path.append("/large/naru/HOGraspNet/src")
from model.classify import TaxonomyClassifier
from dataset.dataset_mano import HOGDataset
from torch.utils.data import DataLoader
import cv2
from omegaconf import OmegaConf
from tqdm import tqdm
import os
cfg = OmegaConf.load("src/cfg/cfg_mano.yaml")
val_dataset = HOGDataset(split="test", db_path=cfg.dataset.db_path, input_type=cfg.model.input_type, mano_type=cfg.model.mano_type, preload=cfg.dataset.preload, noise_level=cfg.train.noise_level)
val_dataloader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=False)
os.makedirs("rgb", exist_ok=True)
for i, batch in tqdm(enumerate(val_dataloader)):
    rgb = batch["img_feat"]
    rgb = rgb.numpy()
    rgb = rgb.transpose(0, 2, 3, 1)
    rgb = rgb[0]
    # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"rgb/rgb_{i+1}.png", rgb)
    print(f"rgb_{i}.png")