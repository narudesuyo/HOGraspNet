import sys 
sys.path.append("/large/naru/HOGraspNet/src")
import pickle
from dataset.HOG_dataloader import HOGDataset
import cv2
import numpy as np
from smplx import MANO, MANOLayer
from hamer_util import *
import trimesh
import torch
from metrics.utils import *
from tqdm import tqdm
sys.path.append("/large/naru/HOGraspNet/")
from thirdparty.manopth.manopth.manolayer import ManoLayer
from collections import defaultdict
dataset = HOGDataset(
    db_path="/large/naru/HOGraspNet/data/",
    setup="s0",
    split="test",
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "/large/naru/InterHandGen/misc/mano_v1_2/models"
mano_model = MANO(model_path=path, use_pca=False, is_rhand=True, flat_hand_mean=True).to(device)
mano_faces = mano_model.faces
mano_layer = ManoLayer(side='right', mano_root=path, use_pca=False, flat_hand_mean=True,
                                center_idx=0, ncomps=45, root_rot_mode="axisang", joint_rot_mode="axisang").to(device)
total_pampjpe_joint = 0.0
total_pampjpe_joint_finetune = 0.0
counts = 0

export_mesh = False
num_samples = 10000
import random
indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
print(f"num_samples: {len(indices)}")
pbar = tqdm(indices, desc="Evaluating", dynamic_ncols=True)
import ast
for i in pbar:
    data = dataset[i]
    img_path = data["rgb_path"]
    img = cv2.imread(img_path)
    mano_path = data["mano_path"]
    occluded_path = img_path.replace("source_augmented", "occluded").replace("/rgb_crop", "").replace("jpg", "txt")
    with open(occluded_path, "r") as f:
        occluded_label = ast.literal_eval(f.read())
    occluded_count = sum(map(int, occluded_label))
    if occluded_count > 15:
        print(f"index: {i}, occluded_count: {occluded_count}, img_path: {img_path}")