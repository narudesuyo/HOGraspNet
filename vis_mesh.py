import os
import numpy as np
import torch
import trimesh
import pyrender
from smplx import MANO
from PIL import Image
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
import sys
sys.path.append('/large/naru/HOGraspNet/src')
sys.path.append("/large/naru/HOGraspNet")
from dataset.HOG_dataloader import HOGDataset
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import trimesh
import pyrender
sys.path.append("/large/naru/HOGraspNet/thirdparty/manopth")
sys.path.append("/large/naru/HOGraspNet/thirdparty/manopth/mano")

from thirdparty.manopth.manopth.manolayer import ManoLayer

def create_bone_cylinder(p1, p2, radius=0.5, color=(0.2, 0.6, 1.0, 1.0)):
    vec = p2 - p1
    height = np.linalg.norm(vec)
    if height < 1e-6:
        return None
    # z軸とvecを合わせる回転行列
    direction = vec / height
    z_axis = np.array([0, 0, 1])
    rotation_matrix = trimesh.geometry.align_vectors(z_axis, direction)
    transform = np.eye(4)
    transform = rotation_matrix.copy()
    transform[:3, 3] = (p1 + p2) / 2  # 中心に配置
    # シリンダー作成と変換
    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=12)
    cylinder.apply_transform(transform)
    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=color)
    return pyrender.Mesh.from_trimesh(cylinder, material=material, smooth=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mano_model_path = '/large/naru/HOGraspNet/thirdparty/mano_v1_2/models/mano'
mano_r_layer = ManoLayer(side='right', mano_root=mano_model_path, use_pca=False, flat_hand_mean=True,
                                center_idx=0, ncomps=45, root_rot_mode="axisang", joint_rot_mode="axisang").to(device)
mano_l_layer = ManoLayer(side='left', mano_root=mano_model_path, use_pca=False, flat_hand_mean=True,
                                center_idx=0, ncomps=45, root_rot_mode="axisang", joint_rot_mode="axisang").to(device)
# mano_r_layer = MANO(model_path=mano_model_path, use_pca=False, is_rhand=True).to(device)
# mano_l_layer = MANO(model_path=mano_model_path, use_pca=False, is_rhand=False).to(device)

def visualize_hand_object_sample(dataloader, idx, mano_model_path):
    os.makedirs("mesh_output", exist_ok=True)
    output_path = f"mesh_output/{idx}.png"
    data = dataloader[idx]
    anno_data = data['anno_data']

    obj_base_name = os.path.splitext(anno_data["Mesh"][0]["object_file"])[0]
    object_file = f"data/obj_scanned_models/{obj_base_name}/{anno_data['Mesh'][0]['object_file']}"
    object_mat = np.array(anno_data["Mesh"][0]["object_mat"])
    mesh = trimesh.load(object_file)

    # MANO parameters
    mano_side = anno_data["Mesh"][0]["mano_side"]
    mano_trans = torch.tensor(anno_data["Mesh"][0]["mano_trans"]).to(device)
    mano_pose = torch.tensor(anno_data["Mesh"][0]["mano_pose"]).to(device)
    mano_betas = torch.tensor(anno_data["Mesh"][0]["mano_betas"]).to(device)

    mano_param = torch.cat([mano_trans, mano_pose], dim=1).to(device)

    mano_layer = mano_r_layer if mano_side == "right" else mano_l_layer
    hand_verts, hand_joints = mano_layer(mano_param, mano_betas)

    # scale and position
    mano_scale = anno_data["hand"]["mano_scale"]
    mano_xyz_root = np.array(anno_data["hand"]["mano_xyz_root"])
    hand_verts = hand_verts.detach().cpu().numpy()/mano_scale + mano_xyz_root
    hand_faces = mano_layer.th_faces.repeat(1, 1, 1).detach().cpu().numpy() if isinstance(mano_layer.th_faces, torch.Tensor) else mano_layer.th_faces.repeat(1, 1, 1).detach().cpu().numpy()
    hand_verts = hand_verts[0]
    hand_faces = hand_faces[0]
    print(f"hand_verts: {hand_verts.shape}, hand_faces: {hand_faces.shape}")
    # contact heatmap coloring
    contact = np.array(anno_data["contact"]).flatten()
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = matplotlib.colormaps['Reds']
    rgba_colors = cmap(norm(contact))
    vertex_colors = (rgba_colors * 255).astype(np.uint8)

    hand_trimesh = trimesh.Trimesh(vertices=hand_verts, faces=hand_faces, vertex_colors=vertex_colors, process=False)
    hand_mesh = pyrender.Mesh.from_trimesh(hand_trimesh, smooth=False)

    # prepare obj mesh
    if isinstance(mesh, trimesh.Trimesh):
        obj_trimesh = mesh.apply_scale(1/1)
    elif isinstance(mesh, trimesh.Scene):
        obj_trimesh = trimesh.util.concatenate(mesh.dump())
    else:
        raise TypeError("Unsupported object mesh type")
    obj_mesh = pyrender.Mesh.from_trimesh(obj_trimesh, smooth=False)

    # pyrender scene
    scene = pyrender.Scene()

    # 3d pose
    joints = np.array(anno_data['hand']['3D_pose_per_cam'])  # shape (21, 3)

    # 各関節を球で可視化
    # for i in range(21):
    #     center = joints[i]
    #     sphere = trimesh.creation.icosphere(radius=1, subdivisions=2)
    #     sphere.apply_translation(center)
    #     joint_mesh = pyrender.Mesh.from_trimesh(sphere, smooth=False)
    #     scene.add(joint_mesh)

    # joint_connections = [
    #     (0, 1), (1, 2), (2, 3), (3, 4),     # Thumb
    #     (0, 5), (5, 6), (6, 7), (7, 8),     # Index
    #     (0, 9), (9,10), (10,11), (11,12),  # Middle
    #     (0,13), (13,14), (14,15), (15,16), # Ring
    #     (0,17), (17,18), (18,19), (19,20)  # Little
    # ]

    # for i, j in joint_connections:
    #     bone = create_bone_cylinder(joints[i], joints[j], radius=0.3, color=(0.8, 0.2, 0.2, 1.0))  # 赤っぽいボーン
    #     if bone:
    #         scene.add(bone)


    scene.add(hand_mesh)
    scene.add(obj_mesh, pose=object_mat)

    # camera, light
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = [0, 0, 120]
    scene.add(camera, pose=cam_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=cam_pose)

    taxonomy_name = anno_data["Mesh"][0]["class_name"]
    object_name = anno_data["object"]["name"]
    caption_text = f"Taxonomy: {taxonomy_name} Object: {object_name}"

    # rendering (pyrender)
    r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
    color, depth = r.render(scene)
    color = color.copy()

    # load as PIL image
    text = caption_text
    text_list = text.split(" ")
    group_size = 12
    grouped_text = [' '.join(text_list[i:i + group_size]) for i in range(0, len(text_list), group_size)]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color_text = (0, 0, 0)  
    thickness = 1
    line_spacing = 20 # line spacing (pixel)

    # start position
    x, y = 10, 30 # position from top left

    # draw each line
    for j, line in enumerate(grouped_text):
        position = (x, y + j * line_spacing)  # shift Y coordinate of each line
        cv2.putText(
            color,
            line,
            position,
            font,
            font_scale,
            color_text,
            thickness,
            cv2.LINE_AA
        )

    # save
    cv2.imwrite(output_path, color)


    r.delete()


if __name__ == "__main__":
    setup = 's2'
    db_path = "/large/naru/HOGraspNet/data"
    split = 'train'
    dataloader = HOGDataset(setup, split, db_path)

    for idx in tqdm(range(len(dataloader))):
        visualize_hand_object_sample(dataloader, idx, mano_model_path)
