import os
import pickle
import torch
from hamer_util import *


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
    device = next(mano_layer.parameters()).device
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

    mano_params = mano_data["pred_mano_params"]
    pose_rot = mano_params["hand_pose"]
    pose_axis = rotation_matrix_to_axis_angle(pose_rot)
    pose_axis = torch.tensor(pose_axis, device=device, dtype=torch.float32).unsqueeze(0)
    shape = torch.tensor(mano_params["betas"], device=device, dtype=torch.float32).unsqueeze(0)

    return pose_axis, shape