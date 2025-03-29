import sys
sys.path.append('/large/naru/HOGraspNet/src')
from dataset.HOG_dataloader import HOGDataset
import os
import smplx
import torch
from tqdm import tqdm
import h5py
import numpy as np
import gc
import psutil
import trimesh
from smplx import MANO

idx = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mano_model_path = '/large/naru/HOGraspNet/thirdparty/mano_v1_2/models/mano'
mano_r_layer = MANO(
    model_path=mano_model_path,
    use_pca=False,
    is_rhand=True  # ← 左手（右手ならTrue）
).to(device)
mano_l_layer = MANO(
    model_path=mano_model_path,
    use_pca=False,
    is_rhand=False  # ← 左手（右手ならTrue）
).to(device)


setup = 's2'
db_path = "/large/naru/HOGraspNet/data"
split = 'train'
dataloader = HOGDataset(setup, split, db_path)
num_samples = len(dataloader)
data = dataloader[idx]
anno_data = data['anno_data']
print(anno_data["Mesh"][0]["object_file"])
obj_base_name = os.path.splitext(anno_data["Mesh"][0]["object_file"])[0]
object_file = f"data/obj_scanned_models/{obj_base_name}/{anno_data['Mesh'][0]['object_file']}"
object_mat = anno_data["Mesh"][0]["object_mat"]
mesh = trimesh.load(object_file)


mano_side = anno_data["Mesh"][0]["mano_side"]
mano_trans = torch.tensor(anno_data["Mesh"][0]["mano_trans"]).to(device)
mano_pose = torch.tensor(anno_data["Mesh"][0]["mano_pose"]).to(device)
mano_betas = torch.tensor(anno_data["Mesh"][0]["mano_betas"]).to(device)
print(f"mano_side: {mano_side}" )
print(f"mano_trans: {mano_trans.shape}")
print(f"mano_pose: {mano_pose.shape}")
print(f"mano_betas: {mano_betas.shape}")
if mano_side == "right":
    mano_layer = mano_r_layer
else:
    mano_layer = mano_l_layer

output = mano_layer(
    hand_pose=mano_pose,
    betas=mano_betas,
    trans=mano_trans
)
import pyrender
from PIL import Image

mano_scale = anno_data["hand"]["mano_scale"]
mano_xyz_root = anno_data["hand"]["mano_xyz_root"]
print(f"mano_scale: {mano_scale}")
print(f"mano_xyz_root: {mano_xyz_root}")

# ========== HAND MESH 用意 ==========
import matplotlib.cm as cm
import matplotlib.colors as colors
contact = np.array(anno_data["contact"]).flatten()  # shape (778,)

# カラーマップ準備（濃淡を表現：0→薄色、1→濃色）
norm = colors.Normalize(vmin=0.0, vmax=1.0)  # contact値を0～1に正規化
cmap = cm.get_cmap('Reds')  # 赤系のグラデーション（'hot', 'plasma', 'viridis'もOK）

# RGBAに変換（0-1 float）
rgba_colors = cmap(norm(contact))  # shape: (778, 4)

# RGBAをuint8に変換（0-255 int）
vertex_colors = (rgba_colors * 255).astype(np.uint8)  # shape: (778, 4)

# 非接触点を薄いグレーにしたい場合（オプション）
# vertex_colors[contact < 0.05] = [180, 180, 180, 255]

# 手メッシュ作成
hand_verts = output.vertices[0].detach().cpu().numpy() * mano_scale * 20 + mano_xyz_root  # (778, 3)
hand_faces = mano_layer.faces.detach().cpu().numpy() if isinstance(mano_layer.faces, torch.Tensor) else mano_layer.faces

hand_trimesh = trimesh.Trimesh(vertices=hand_verts, faces=hand_faces, vertex_colors=vertex_colors, process=False)
hand_mesh = pyrender.Mesh.from_trimesh(hand_trimesh, smooth=False)

# ========== OBJECT MESH 用意 ==========
if isinstance(mesh, trimesh.Trimesh):
    obj_trimesh = mesh.apply_scale(1/1)
    print("hihiii")
elif isinstance(mesh, trimesh.Scene):
    obj_trimesh = trimesh.util.concatenate(mesh.dump())  # 複数パーツ結合
else:
    raise TypeError("Unsupported mesh type")

obj_mesh = pyrender.Mesh.from_trimesh(obj_trimesh, smooth=False)

# ========== pyrender シーン作成 ==========
scene = pyrender.Scene()
scene.add(hand_mesh)
scene.add(obj_mesh, pose=object_mat)
print(f"object_mat: {object_mat}")
# ========== カメラ設定 ==========
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
cam_pose = np.eye(4)
cam_pose[:3, 3] = [0, 0, 120]  # カメラを前方に
scene.add(camera, pose=cam_pose)

# ========== ライト追加 ==========
light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
scene.add(light, pose=cam_pose)

# ========== レンダリングと保存 ==========
r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
color, depth = r.render(scene)

# 保存
Image.fromarray(color).save("hand_obj_render.png")
print("✅ 画像保存しました → hand_obj_render.png")

r.delete()
taxonomy = data["taxonomy"]
print(f"taxonomy: {taxonomy}")



