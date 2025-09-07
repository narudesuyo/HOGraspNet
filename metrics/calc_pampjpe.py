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
from tqdm import tqdm
sys.path.append("/large/naru/HOGraspNet/")
from optimization.util import *
from thirdparty.manopth.manopth.manolayer import ManoLayer
from collections import defaultdict
import ast
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "/large/naru/InterHandGen/misc/mano_v1_2/models"
mano_layer = ManoLayer(side='right', mano_root=path, use_pca=False, flat_hand_mean=True,
                                center_idx=0, ncomps=45, root_rot_mode="axisang", joint_rot_mode="axisang").to(device)
def compute_pampjpe_from_params(gt_pose, gt_shape, pred_pose, pred_shape, mano_layer):
    """
    引数:
        gt_pose: torch.Tensor [45] or [1, 45]
        gt_shape: torch.Tensor [10] or [1, 10]
        pred_pose: torch.Tensor [45] or [1, 45]
        pred_shape: torch.Tensor [10] or [1, 10]
        mano_model: MANOモデル（hand_pose, betas を受け取る call に対応）
    
    戻り値:
        pampjpe: float (単位: mm)
    """
    # shape調整
    gt_pose = gt_pose.view(1, 45)
    gt_shape = gt_shape.view(1, 10)
    pred_pose = pred_pose.view(1, 45)
    pred_shape = pred_shape.view(1, 10)

    # GPU対応
    device = next(mano_model.parameters()).device
    gt_pose = gt_pose.to(device)
    gt_shape = gt_shape.to(device)
    pred_pose = pred_pose.to(device)
    pred_shape = pred_shape.to(device)

    # MANOで3D jointを取得
    gt_verts, gt_joints = mano_layer(torch.cat([torch.zeros(1, 3).to(device), gt_pose], dim=1), gt_shape)
    pred_verts, pred_joints = mano_layer(torch.cat([torch.zeros(1, 3).to(device), pred_pose], dim=1), pred_shape)

    gt_joints = gt_joints.detach().cpu().numpy()  # shape: [21, 3]
    pred_joints = pred_joints.detach().cpu().numpy()  # shape: [21, 3]

    # root joint (通常0番目) を原点に平行移動
    gt_joints -= gt_joints[0][0]
    pred_joints -= pred_joints[0][0]

    pampjpe = reconstruction_error(gt_joints, pred_joints)
    return pampjpe
def get_joints_from_mano_path(path, mano_layer, device):
    """
    MANOパラメータのpickleファイルパスを与えると、対応するjoint座標を返す。
    
    引数:
        path: str, MANOパラメータのpickleファイルパス
        mano_layer: ManoLayer オブジェクト
        device: torch.device
    
    戻り値:
        joints: np.ndarray [21, 3] or None（ファイルが存在しない場合）
    """
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return None

    with open(path, "rb") as f:
        mano_data = pickle.load(f)

    if type(mano_data) == list:
        pose_axis_list = []
        shape_list = []
        for i in range(len(mano_data)):
            mano_params = mano_data[i]["pred_mano_params"]
            pose_rot = mano_params["hand_pose"]
            pose_axis = rotation_matrix_to_axis_angle(pose_rot)
            pose_axis = torch.tensor(pose_axis, device=device, dtype=torch.float32).unsqueeze(0)
            shape = torch.tensor(mano_params["betas"], device=device, dtype=torch.float32).unsqueeze(0)
            pose_axis_list.append(pose_axis.squeeze(0))
            shape_list.append(shape.squeeze(0))
        return torch.cat(pose_axis_list, dim=0), torch.cat(shape_list, dim=0)
    else:
        mano_params = mano_data["pred_mano_params"]
        pose_rot = mano_params["hand_pose"]
        pose_axis = rotation_matrix_to_axis_angle(pose_rot)
        pose_axis = torch.tensor(pose_axis, device=device, dtype=torch.float32).unsqueeze(0)
        shape = torch.tensor(mano_params["betas"], device=device, dtype=torch.float32).unsqueeze(0)
        return pose_axis, shape

setup = "s1"
# calc_setups = ["zero", "wilor", "s0"]
calc_setups = ["zero", "wilor", "hamba"]
# split = ["train", "test"]
split = ["train"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "/large/naru/InterHandGen/misc/mano_v1_2/models"
mano_model = MANO(model_path=path, use_pca=False, is_rhand=True, flat_hand_mean=True).to(device)
mano_faces = mano_model.faces
mano_layer = ManoLayer(side='right', mano_root=path, use_pca=False, flat_hand_mean=True,
                                center_idx=0, ncomps=45, root_rot_mode="axisang", joint_rot_mode="axisang").to(device)
import random
# sample_idx = range(len(dataset)-1)

# sample_idx = np.random.choice(sample_idx, len(sample_idx), replace=False)
occluded_count_zero = defaultdict(list)
occluded_count_s0 = defaultdict(list)
occluded_count_s2 = defaultdict(list)
occluded_count_s3 = defaultdict(list)
occluded_count_wilor = defaultdict(list)  
occluded_count_hamba = defaultdict(list)  
total_pampjpe_zero = 0
total_pampjpe_s0 = 0
total_pampjpe_s2 = 0
total_pampjpe_s3 = 0
total_pampjpe_wilor = 0
total_pampjpe_hamba = 0
sample_count = 0
for s in split:
    dataset = HOGDataset(
        db_path="/large/naru/HOGraspNet/data_selected/",
        setup=setup,
        split=s,
        # load_pkl=False,
    )
    # for i in tqdm(range(len(dataset))):
    for i in tqdm(range(400)):
        data = dataset[i]
        img_path = data["rgb_path"]

        # 必要な全てのパスを先に組み立てて確認
        required_paths = []
        if "s0" in calc_setups:
            required_paths.append(
                img_path.replace("source_augmented", "s0_hamer").replace("jpg", "pkl").replace("/rgb_crop", "")
            )
        if "zero" in calc_setups:
            required_paths.append(
                img_path.replace("rgb_crop/", "").replace("jpg", "pkl").replace("/source_augmented", "/zero_shot_hamer")
            )
        if "wilor" in calc_setups:
            required_paths.append(
                img_path.replace("source_augmented", "mano_wilor").replace("jpg", "pkl").replace("/rgb_crop", "")
            )
        if "s2" in calc_setups:
            required_paths.append(
                data["finetune_mano_path"].replace("finetune_mano","mano_s2")
            )
        if "s3" in calc_setups:
            required_paths.append(
                data["finetune_mano_path"].replace("finetune_mano","mano_s3")
            )
        if "hamba" in calc_setups:
            required_paths.append(
                img_path.replace("source_augmented", "mano_hamba").replace("jpg", "pkl").replace("/rgb_crop", "")
            )
        # どれか一つでも存在しなければスキップ
        if not all(os.path.exists(p) for p in required_paths):
            continue
        sample_count += 1
        # with open(occluded_path, "r") as f:
        #     occluded_label = ast.literal_eval(f.read())
        # occluded_count = sum(map(int, occluded_label))
        mano_path = data["mano_path"]

        gt_pose = torch.tensor(data["mano_pose"]).to(device)
        gt_shape = torch.tensor(data["mano_betas"]).to(device)
        if "zero" in calc_setups:
            # zero_mano_path = data["mano_path"]
            zero_mano_path = data["rgb_path"].replace("rgb_crop/", "").replace("jpg", "pkl").replace("/source_augmented", "/zero_shot_hamer")
            zero_mano_pose_axis, zero_mano_shape = get_joints_from_mano_path(zero_mano_path, mano_layer, device)
            pampjpe_zero = compute_pampjpe_from_params(gt_pose, gt_shape, zero_mano_pose_axis, zero_mano_shape, mano_layer)
            total_pampjpe_zero += pampjpe_zero.item()
            # occluded_count_zero[occluded_count].append(pampjpe_zero.item())
            print(f"pampjpe_zero: {pampjpe_zero.item():.2f}")
        if "wilor" in calc_setups:
            wilor_mano_path = img_path.replace("source_augmented", "mano_wilor").replace("jpg", "pkl").replace("/rgb_crop", "")
            wilor_mano_pose_axis, wilor_mano_shape = get_joints_from_mano_path(wilor_mano_path, mano_layer, device)
            if wilor_mano_shape.shape[1] != 1:
                best_pampjpe_wilor = float("inf")
                for j in range(wilor_mano_shape.shape[1]):
                    wilor_shape_j = wilor_mano_shape[0][j][:]
                    wilor_pose_axis_j = wilor_mano_pose_axis[0][j][:]
                    pampjpe_wilor_j = compute_pampjpe_from_params(gt_pose, gt_shape, wilor_pose_axis_j, wilor_shape_j, mano_layer)
                    if pampjpe_wilor_j.item() < best_pampjpe_wilor:
                        best_pampjpe_wilor = pampjpe_wilor_j
                pampjpe_wilor = best_pampjpe_wilor
                total_pampjpe_wilor += best_pampjpe_wilor.item()
            else:
                pampjpe_wilor = compute_pampjpe_from_params(gt_pose, gt_shape, wilor_mano_pose_axis, wilor_mano_shape, mano_layer)
                total_pampjpe_wilor += pampjpe_wilor.item()
            print(f"pampjpe_wilor: {pampjpe_wilor.item():.2f}")
        if "s0" in calc_setups:
            s0_mano_path = img_path.replace("source_augmented", "s0_hamer").replace("jpg", "pkl").replace("/rgb_crop", "")
            s0_mano_pose_axis, s0_mano_shape = get_joints_from_mano_path(s0_mano_path, mano_layer, device)
            pampjpe_s0 = compute_pampjpe_from_params(gt_pose, gt_shape, s0_mano_pose_axis, s0_mano_shape, mano_layer)
            total_pampjpe_s0 += pampjpe_s0.item()
            # occluded_count_s0[occluded_count].append(pampjpe_s0.item())
            print(f"pampjpe_s0: {pampjpe_s0.item():.2f}")
        if "s2" in calc_setups:
            s2_mano_path = data["finetune_mano_path"].replace("finetune_mano","mano_s2")
            s2_mano_pose_axis, s2_mano_shape = get_joints_from_mano_path(s2_mano_path, mano_layer, device)
            pampjpe_s2 = compute_pampjpe_from_params(gt_pose, gt_shape, s2_mano_pose_axis, s2_mano_shape, mano_layer)
            total_pampjpe_s2 += pampjpe_s2.item()
            # occluded_count_s2[occluded_count].append(pampjpe_s2.item())
            print(f"pampjpe_s2: {pampjpe_s2.item():.2f}")
        if "s3" in calc_setups:
            s3_mano_path = data["finetune_mano_path"].replace("finetune_mano","mano_s3")
            s3_mano_pose_axis, s3_mano_shape = get_joints_from_mano_path(s3_mano_path, mano_layer, device)
            pampjpe_s3 = compute_pampjpe_from_params(gt_pose, gt_shape, s3_mano_pose_axis, s3_mano_shape, mano_layer)
            total_pampjpe_s3 += pampjpe_s3.item()
            # occluded_count_s3[occluded_count].append(pampjpe_s3.item())
            print(f"pampjpe_s3: {pampjpe_s3.item():.2f}")   
        if "hamba" in calc_setups:
            hamba_mano_path = img_path.replace("source_augmented", "mano_hamba").replace("jpg", "pkl").replace("/rgb_crop", "")
            hamba_mano_pose_axis, hamba_mano_shape = get_joints_from_mano_path(hamba_mano_path, mano_layer, device)
            if hamba_mano_shape.shape[0] != 1:
                best_pampjpe_hamba = float("inf")
                for j in range(hamba_mano_shape.shape[0]):
                    hamba_shape_j = hamba_mano_shape[j][:]
                    hamba_pose_axis_j = hamba_mano_pose_axis[j][:]
                    pampjpe_hamba_j = compute_pampjpe_from_params(gt_pose, gt_shape, hamba_pose_axis_j, hamba_shape_j, mano_layer)
                    # print(f"pampjpe_hamba_j: {pampjpe_hamba_j.item():.2f}")
                    if pampjpe_hamba_j.item() < best_pampjpe_hamba:
                        best_pampjpe_hamba = pampjpe_hamba_j
                pampjpe_hamba = best_pampjpe_hamba
            else:
                pampjpe_hamba = compute_pampjpe_from_params(gt_pose, gt_shape, hamba_mano_pose_axis, hamba_mano_shape, mano_layer)
            total_pampjpe_hamba += pampjpe_hamba.item()
            print(f"pampjpe_hamba: {pampjpe_hamba.item():.2f}")
    # for occluded_count in occluded_count_zero.keys():
    #     if "zero" in calc_setups:
    #         print(f"occluded_count_zero: {occluded_count} {np.mean(occluded_count_zero[occluded_count]):.2f}")
    #     if "s0" in calc_setups:
    #         print(f"occluded_count_s0: {occluded_count} {np.mean(occluded_count_s0[occluded_count]):.2f}")
    #     if "s2" in calc_setups:
    #         print(f"occluded_count_s2: {occluded_count} {np.mean(occluded_count_s2[occluded_count]):.2f}")
    #     if "s3" in calc_setups:
    #         print(f"occluded_count_s3: {occluded_count} {np.mean(occluded_count_s3[occluded_count]):.2f}")
    #     if "wilor" in calc_setups:
    #         print(f"occluded_count_wilor: {occluded_count} {np.mean(occluded_count_wilor[occluded_count]):.2f}")
    # with open("metrics/occluded_count.txt", "w") as f:
    #     for occluded_count in occluded_count_zero.keys():
    #         f.write(f"{occluded_count} : zero {np.mean(occluded_count_zero[occluded_count]):.2f} s0 {np.mean(occluded_count_s0[occluded_count]):.2f} s2 {np.mean(occluded_count_s2[occluded_count]):.2f} s3 {np.mean(occluded_count_s3[occluded_count]):.2f} wilor {np.mean(occluded_count_wilor[occluded_count]):.2f}\n")

    if "zero" in calc_setups:
        print(f"total_pampjpe_zero: {total_pampjpe_zero / sample_count:.2f}")
    if "s0" in calc_setups:
        print(f"total_pampjpe_s0: {total_pampjpe_s0 / sample_count:.2f}")
    if "s2" in calc_setups:
        print(f"total_pampjpe_s2: {total_pampjpe_s2 / sample_count:.2f}")
    if "s3" in calc_setups:
        print(f"total_pampjpe_s3: {total_pampjpe_s3 / sample_count:.2f}")
    if "wilor" in calc_setups:
        print(f"total_pampjpe_wilor: {total_pampjpe_wilor / sample_count:.2f}")
    if "hamba" in calc_setups:
        print(f"total_pampjpe_hamba: {total_pampjpe_hamba / sample_count:.2f}")



    # with open("metrics/occluded_count_sample.txt", "w") as f:
    #     for occluded_count in occluded_count_zero.keys():
    #         count = len(occluded_count_zero[occluded_count])
    #         f.write(f"{occluded_count} ({count} samples) : zero {np.mean(occluded_count_zero[occluded_count]):.2f} s0 {np.mean(occluded_count_s0[occluded_count]):.2f} s2 {np.mean(occluded_count_s2[occluded_count]):.2f} wilor {np.mean(occluded_count_wilor[occluded_count]):.2f}\n")

