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
taxonomy_errors_orig = defaultdict(list)
object_errors_orig = defaultdict(list)
taxonomy_errors_finetune = defaultdict(list)
object_errors_finetune = defaultdict(list)
occluded_count_orig = defaultdict(list)
occluded_count_finetune = defaultdict(list)

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
num_samples = 100
import random
indices = list(range(len(dataset)))
indices = random.sample(indices, num_samples)
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
    if occluded_count > 16:
        print(f"Occluded count: {occluded_count}, {img_path}")
    with open(mano_path, "rb") as f:
        mano_data = pickle.load(f)
    mano_params = mano_data["pred_mano_params"]
    pose_rot = mano_params["hand_pose"]
    pose_axis = rotation_matrix_to_axis_angle(pose_rot)
    shape = mano_params["betas"]
    pred_vertices = mano_data["pred_vertices"]
    global_orient_rot = mano_params["global_orient"]
    global_orient_axis = rotation_matrix_to_axis_angle(global_orient_rot)
    focal_length = mano_data["focal_length"]
    cam_t = mano_data["pred_cam_t"]  # ← ここたぶん mano_params ではなく mano_data
    finetune_mano_path = data["finetune_mano_path"].replace("finetune_mano", "mano_finetune")
    if not os.path.exists(finetune_mano_path):
        print(f"Finetuned MANO file not found for {i}")
        continue

    with open(finetune_mano_path, "rb") as f:
        finetune_mano_data = pickle.load(f)
    finetune_mano_params = finetune_mano_data["pred_mano_params"]
    finetune_pose_rot = finetune_mano_params["hand_pose"]
    finetune_pose_axis = rotation_matrix_to_axis_angle(finetune_pose_rot.unsqueeze(0))
    finetune_shape = finetune_mano_params["betas"]
    finetune_global_orient_rot = finetune_mano_params["global_orient"]
    finetune_global_orient_axis = rotation_matrix_to_axis_angle(finetune_global_orient_rot)
    gt_pose = torch.tensor(data["mano_pose"]).to(device)
    gt_shape = torch.tensor(data["mano_betas"]).to(device)
    gt_trans = torch.tensor(data["mano_trans"]).to(device)
    gt_cam = torch.tensor(data["extrinsics"]).to(device)


    output = mano_model(
        hand_pose=pose_axis.to(device).reshape(1,45),
        betas=shape.to(device).reshape(1,10),
        global_orient=global_orient_axis.to(device).reshape(1,3),
        transl=cam_t.to(device).reshape(1,3)
    )
    verts_layer_pred, joint_layer_pred = mano_layer(
        torch.cat([global_orient_axis, pose_axis],dim=1).to(device),
        shape.to(device)
    )
    mesh = trimesh.Trimesh(vertices=verts_layer_pred.squeeze(0).detach().cpu().numpy(), faces=mano_faces, process=False)
    if export_mesh:
        save_path = f"./metrics/meshes/{i}_orig.ply"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mesh.export(save_path)
    joint_layer_pred_np = joint_layer_pred.detach().cpu().numpy()+cam_t.to(device).reshape(1,3).detach().cpu().numpy()
    verts_layer_pred_np = verts_layer_pred.detach().cpu().numpy()+ cam_t.to(device).reshape(1,3).detach().cpu().numpy()

    joint_pred = output.joints
    joint_pred_np = joint_pred.detach().cpu().numpy()
    verts = output.vertices
    verts_np = verts.detach().cpu().numpy()
    verts_layer_finetune, joint_layer_finetune = mano_layer(
        torch.cat([finetune_global_orient_axis, finetune_pose_axis],dim=1).to(device),
        finetune_shape.to(device)
    )
    mesh = trimesh.Trimesh(vertices=verts_layer_finetune.squeeze(0).detach().cpu().numpy(), faces=mano_faces, process=False)
    if export_mesh:
        save_path = f"./metrics/meshes/{i}_finetune.ply"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mesh.export(save_path)


    joint_layer_finetune_np = joint_layer_finetune.detach().cpu().numpy()
    verts_layer_finetune_np = verts_layer_finetune.detach().cpu().numpy()

    output_gt = mano_model(
        hand_pose=gt_pose.to(device).reshape(1,45),
        betas=gt_shape.to(device).reshape(1,10),
        global_orient=gt_trans.to(device).reshape(1,3),
        transl=torch.zeros(1,3).to(device)
    )
    verts_layer_gt, joint_layer_gt = mano_layer(
        torch.cat([gt_trans, gt_pose],dim=1).to(device),
        gt_shape.to(device)
    )
    mesh = trimesh.Trimesh(vertices=verts_layer_gt.squeeze(0).detach().cpu().numpy(), faces=mano_faces, process=False)
    if export_mesh:
        save_path = f"./metrics/meshes/{i}_gt.ply"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mesh.export(save_path)

    joint_layer_gt = joint_layer_gt.detach().cpu().numpy()

    joint_gt = output_gt.joints
    joint_gt_np = joint_gt.detach().cpu().numpy()
    verts_gt = output_gt.vertices
    verts_gt_np = verts_gt.detach().cpu().numpy()
    hand_xyz_root = np.array(data["mano_xyz_root"])
    scale = data["mano_scale"]

    joint_layer_gt_np = joint_layer_gt + hand_xyz_root*scale
    joint_gt_np = joint_gt_np + hand_xyz_root*scale

    extrinsics = data["extrinsics"]
    extrinsics = extrinsics
    joint_layer_gt_np = mano3DToCam3D(torch.from_numpy(joint_layer_gt_np).to(device), extrinsics).cpu().numpy()
    joint_layer_gt_np = joint_layer_gt_np
    joint_layer_pred_np = joint_layer_pred_np
    joint_layer_gt_np = np.expand_dims(joint_layer_gt_np, axis=0)
    mpjpe_before = mpjpe(joint_layer_pred_np, joint_layer_gt_np)
    pampjpe_before = reconstruction_error(joint_layer_gt_np, joint_layer_pred_np)
    joint_layer_gt_np = joint_layer_gt_np - joint_layer_gt_np[0][0]
    joint_layer_pred_np = joint_layer_pred_np - joint_layer_pred_np[0][0]
    mpjpe_joint = mpjpe(joint_layer_pred_np, joint_layer_gt_np)
    pampjpe_joint = reconstruction_error(joint_layer_gt_np, joint_layer_pred_np)
    pampjpe_joint_finetune = reconstruction_error(joint_layer_gt_np, joint_layer_finetune_np)
    # print(f"pampjpe_joint: {pampjpe_joint.item():.2f}, pampjpe_joint_finetune: {pampjpe_joint_finetune.item():.2f}")
    total_pampjpe_joint += pampjpe_joint.item()
    total_pampjpe_joint_finetune += pampjpe_joint_finetune.item()
    counts += 1
    taxonomy_errors_orig[data["taxonomy_id"]].append(pampjpe_joint.item())
    object_errors_orig[data["object_id"]].append(pampjpe_joint.item())
    taxonomy_errors_finetune[data["taxonomy_id"]].append(pampjpe_joint_finetune.item())
    object_errors_finetune[data["object_id"]].append(pampjpe_joint_finetune.item())
    occluded_count_orig[occluded_count].append(pampjpe_joint.item())
    occluded_count_finetune[occluded_count].append(pampjpe_joint_finetune.item())
    avg_joint = total_pampjpe_joint / counts
    avg_joint_finetune = total_pampjpe_joint_finetune / counts

    pbar.set_postfix({
        "avg_pampjpe": f"{avg_joint:.2f}",
        "avg_finetune": f"{avg_joint_finetune:.2f}"
    })
    # if i > 500:
    #     break

with open("./metrics/taxonomy_object_pampjpe.txt", "w") as f:
    # Taxonomy-wise
    f.write("--- Taxonomy-wise PAMPJPE ---\n")
    all_tax = sorted(set(taxonomy_errors_orig.keys()) | set(taxonomy_errors_finetune.keys()))
    for tax in all_tax:
        orig_vals = taxonomy_errors_orig.get(tax, [])
        finetune_vals = taxonomy_errors_finetune.get(tax, [])
        orig_avg = sum(orig_vals) / len(orig_vals) if orig_vals else None
        finetune_avg = sum(finetune_vals) / len(finetune_vals) if finetune_vals else None
        sample_count = max(len(orig_vals), len(finetune_vals))

        if orig_avg is not None and finetune_avg is not None:
            diff = finetune_avg - orig_avg
            f.write(f"{tax}: {orig_avg:.2f} → {finetune_avg:.2f} ({diff:+.2f}) mm | samples: {sample_count}\n")
        elif orig_avg is not None:
            f.write(f"{tax}: {orig_avg:.2f} → N/A mm | samples: {sample_count}\n")
        elif finetune_avg is not None:
            f.write(f"{tax}: N/A → {finetune_avg:.2f} mm | samples: {sample_count}\n")

    # Object-wise
    f.write("\n--- Object-wise PAMPJPE ---\n")
    all_obj = sorted(set(object_errors_orig.keys()) | set(object_errors_finetune.keys()))
    for obj in all_obj:
        orig_vals = object_errors_orig.get(obj, [])
        finetune_vals = object_errors_finetune.get(obj, [])
        orig_avg = sum(orig_vals) / len(orig_vals) if orig_vals else None
        finetune_avg = sum(finetune_vals) / len(finetune_vals) if finetune_vals else None
        sample_count = max(len(orig_vals), len(finetune_vals))

        if orig_avg is not None and finetune_avg is not None:
            diff = finetune_avg - orig_avg
            f.write(f"{obj}: {orig_avg:.2f} → {finetune_avg:.2f} ({diff:+.2f}) mm | samples: {sample_count}\n")
        elif orig_avg is not None:
            f.write(f"{obj}: {orig_avg:.2f} → N/A mm | samples: {sample_count}\n")
        elif finetune_avg is not None:
            f.write(f"{obj}: N/A → {finetune_avg:.2f} mm | samples: {sample_count}\n")

    # Occluded count-wise
    f.write("\n--- Occluded Count-wise PAMPJPE ---\n")
    all_counts = sorted(set(occluded_count_orig.keys()) | set(occluded_count_finetune.keys()))
    for count in all_counts:
        orig_vals = occluded_count_orig.get(count, [])
        finetune_vals = occluded_count_finetune.get(count, [])
        orig_avg = sum(orig_vals) / len(orig_vals) if orig_vals else None
        finetune_avg = sum(finetune_vals) / len(finetune_vals) if finetune_vals else None
        sample_count = max(len(orig_vals), len(finetune_vals))

        if orig_avg is not None and finetune_avg is not None:
            diff = finetune_avg - orig_avg
            f.write(f"{count}: {orig_avg:.2f} → {finetune_avg:.2f} ({diff:+.2f}) mm | samples: {sample_count}\n")
        elif orig_avg is not None:
            f.write(f"{count}: {orig_avg:.2f} → N/A mm | samples: {sample_count}\n")
        elif finetune_avg is not None:
            f.write(f"{count}: N/A → {finetune_avg:.2f} mm | samples: {sample_count}\n")

    # Overall
    f.write("\n--- Overall Average ---\n")
    f.write(f"Original Avg PAMPJPE: {total_pampjpe_joint / counts:.2f} mm\n")
    f.write(f"Finetuned Avg PAMPJPE: {total_pampjpe_joint_finetune / counts:.2f} mm\n")
    f.write(f"Total Samples: {counts}\n")