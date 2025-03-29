from src.data.HOG_dataloader import HOGDataset
import os
setup = 's2'
split = 'test'
db_path = "/large/naru/HOGraspNet/data"
dataloader = HOGDataset(setup, split, db_path)
print(len(dataloader))
import smplx
import torch

r_mano_layer = smplx.create(
    model_path='/large/naru/HOGraspNet/thirdparty/mano_v1_2/models',
    model_type='mano',
    use_pca=False,
    is_rhand=True,
    batch_size=1,
    gender='neutral'
)
l_mano_layer = smplx.create(
    model_path='/large/naru/HOGraspNet/thirdparty/mano_v1_2/models',
    model_type='mano',
    use_pca=False,
    is_rhand=False,
    batch_size=1,
    gender='neutral'
)

for i in range(len(dataloader)):
    data = dataloader[i]
    anno_data = data['anno_data']
    print(anno_data['Mesh'][0]["class_id"])
    mano_trans = torch.tensor(anno_data['Mesh'][0]["mano_trans"])
    mano_pose = torch.tensor(anno_data['Mesh'][0]["mano_pose"])
    mano_betas = torch.tensor(anno_data['Mesh'][0]["mano_betas"])
    mano_side = anno_data['Mesh'][0]["mano_side"]
    print(mano_side)
    print(mano_trans.shape)
    print(mano_pose.shape)
    print(mano_betas.shape)
    if mano_side == "right":
        mano_output = r_mano_layer(
            global_orient=torch.zeros(1, 3),
            hand_pose=mano_pose,
            betas=mano_betas,
            transl=mano_trans
        )
    else:
        mano_output = l_mano_layer(
            global_orient=torch.zeros(1, 3),
            hand_pose=mano_pose,
            betas=mano_betas,
            transl=mano_trans
        )   
    # 出力の確認
    vertices = mano_output.vertices
    print("頂点の形状:", vertices.shape)  # 期待値: (1, 778, 3)
    break
