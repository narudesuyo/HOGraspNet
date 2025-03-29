import pickle
from smplx import MANO
import torch
import trimesh
import pyrender
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

# ディレクトリ内のpickleファイルを取得
pickle_dir = 'Hand_pose/handpose_left_hand/ZY20210800001/H1/C20/N18/S286/s01/T1'
pickle_files = sorted([f for f in os.listdir(pickle_dir) if f.endswith('.pickle')])

# 各pickleファイルを処理
for i, pickle_file in tqdm(enumerate(pickle_files), total=len(pickle_files)):
    with open(os.path.join(pickle_dir, pickle_file), 'rb') as f:
        data = pickle.load(f)

    pose_coeff = torch.from_numpy(data['poseCoeff'])
    shape_coeff = torch.from_numpy(data['beta'])
    trans = torch.from_numpy(data['trans'])

    mano_layer = MANO(
        model_path='thirdparty/mano_v1_2/models/mano',   # MANOモデルファイル（*.pkl）へのパス
        use_pca=False,                      # PCAを使わず、48次元のfull poseで渡す場合
        is_rhand=False,
        batch_size=1                     # 右手モデルか左手モデルか
    )

    output = mano_layer(
        global_orient=pose_coeff[:3].unsqueeze(0).float(),    # 手首の回転（root orientation）
        hand_pose=pose_coeff[3:].unsqueeze(0).float(),        # 指の回転（45次元）
        betas=shape_coeff.unsqueeze(0).float(),                  # 手の形状（ベータ）
        transl=trans.unsqueeze(0).float()                        # 手の平行移動
    )

    vertices = output.vertices.cpu().numpy()     # 手のメッシュの頂点座標（[1, 778, 3]）
    joints = output.joints
    faces = mano_layer.faces
    mesh = trimesh.Trimesh(vertices=vertices[0], faces=faces, process=False)

    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh_pyrender)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = [-0.2, 0.3, 1.5]  # カメラを少し前に置く
    scene.add(camera, pose=cam_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=cam_pose)

    r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
    color, depth = r.render(scene)

    output_image_path = f"output_images/frame_{i:04d}.png"
    Image.fromarray(color).save(output_image_path)

# with open('Hand_pose/handpose_left_hand/ZY20210800001/H1/C20/N18/S286/s01/T1/30.pickle', 'rb') as f:
#     data = pickle.load(f)

# for key in data.keys():
#     print(f"{key}: {data[key].shape}")
# pose_coeff = torch.from_numpy(data['poseCoeff'])
# shape_coeff = torch.from_numpy(data['beta'])
# trans = torch.from_numpy(data['trans'])

# mano_layer = MANO(
#     model_path='thirdparty/mano_v1_2/models/mano',   # MANOモデルファイル（*.pkl）へのパス
#     use_pca=False,                      # PCAを使わず、48次元のfull poseで渡す場合
#     is_rhand=False,
#     batch_size=1                     # 右手モデルか左手モデルか
# )

# output = mano_layer(
#     global_orient=pose_coeff[:3].unsqueeze(0).float(),    # 手首の回転（root orientation）
#     hand_pose=pose_coeff[3:].unsqueeze(0).float(),        # 指の回転（45次元）
#     betas=shape_coeff.unsqueeze(0).float(),                  # 手の形状（ベータ）
#     transl=trans.unsqueeze(0).float()                        # 手の平行移動
# )

# # 出力内容
# vertices = output.vertices.cpu().numpy()     # 手のメッシュの頂点座標（[1, 778, 3]）
# joints = output.joints
# faces = mano_layer.faces
# print(f"joints.shape: {joints.shape}")
# print(f"vertices.shape: {vertices.shape}")
# print(f"faces.shape: {faces.shape}")
# mesh = trimesh.Trimesh(vertices=vertices[0], faces=faces, process=False)

# mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
# scene = pyrender.Scene()
# scene.add(mesh_pyrender)

# camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
# cam_pose = np.eye(4)
# cam_pose[:3, 3] = [-0.2, 0.4, 1]  # カメラを少し前に置く
# scene.add(camera, pose=cam_pose)

# light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
# scene.add(light, pose=cam_pose)

# r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
# color, depth = r.render(scene)

# Image.fromarray(color).save("output_images/output_hand_render2.png")