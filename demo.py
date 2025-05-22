from src.dataset.HOG_dataloader import HOGDataset
import os
from tqdm import tqdm
import h5py
import os
import pickle
from tqdm import tqdm
import numpy as np
import torch

import numpy as np
import gc

def select_nearest_hand_obj_pair(bbox):
    """
    bboxから最も近い手とオブジェクトのペアを選ぶ。

    Args:
        bbox (dict): 'hand_dets' (M, 10) と 'obj_dets' (N, 10) を含む辞書

    Returns:
        selected_hand_dets (np.ndarray): (1, 10) shape の手の検出
        selected_obj_dets (np.ndarray): (1, 10) shape のオブジェクト検出
    """
    obj_dets_all = bbox['obj_dets']  # (N, 10)
    hand_dets_all = bbox['hand_dets']  # (M, 10)

    # オブジェクト中心座標 (N, 2)
    obj_centers = np.stack([
        (obj_dets_all[:, 0] + obj_dets_all[:, 2]) / 2,
        (obj_dets_all[:, 1] + obj_dets_all[:, 3]) / 2
    ], axis=1)

    # 手の中心座標 (M, 2)
    hand_centers = np.stack([
        (hand_dets_all[:, 0] + hand_dets_all[:, 2]) / 2,
        (hand_dets_all[:, 1] + hand_dets_all[:, 3]) / 2
    ], axis=1)

    # 距離計算 (M, N)
    dists = np.linalg.norm(hand_centers[:, None, :] - obj_centers[None, :, :], axis=2)

    # 最小距離のペアを取得
    min_idx = np.unravel_index(np.argmin(dists), dists.shape)
    hand_idx, obj_idx = min_idx

    # 選ばれた手とオブジェクトをexpand_dimsして返す
    selected_hand_dets = np.expand_dims(hand_dets_all[hand_idx], axis=0)  # (1, 4)
    selected_obj_dets = np.expand_dims(obj_dets_all[obj_idx], axis=0)     # (1, 10)

    return selected_hand_dets, selected_obj_dets

# ① Datasetロード
split = 'train'
setup = 's3'
use_aug = True
dataset = HOGDataset(
    setup=setup,
    split=split,
    db_path='/large/naru/HOGraspNet/data',
    load_pkl=False,
    use_aug=use_aug,
)

# ② h5ファイル作成
h5_path = f'/large/naru/HOGraspNet/data/h5_{setup}_vertices_aug/{split}.h5'
os.makedirs(os.path.dirname(h5_path), exist_ok=True)
h5_file = h5py.File(h5_path, 'w')
num_samples = len(dataset)

# valid_samples = []
# for idx in tqdm(range(num_samples), desc="Counting valid samples"):
#     sample = dataset[idx]
#     if sample['bbox']['hand_dets'] is None or sample['bbox']['obj_dets'] is None:
#         continue
#     valid_samples.append(idx)

# print(f"Valid samples: {len(valid_samples)} / {num_samples}")

# print(f"Saving {num_samples} samples into {h5_path}")
# valid_samples = 77656
# valid_samples = 45327

rgb_ds = h5_file.create_dataset('rgb', shape=(num_samples, 480, 640, 3), dtype='uint8', chunks=(64,480,640,3))
taxonomy_ds = h5_file.create_dataset('taxonomy_id', shape=(num_samples,), dtype='int', chunks=(64,))
object_ds = h5_file.create_dataset('object_id', shape=(num_samples,), dtype='int', chunks=(64,))
mano_ds = h5_file.create_dataset('mano_vertices_pred', shape=(num_samples,1,778,3), dtype='float32', chunks=(64,1,778,3))
mano_trans_ds = h5_file.create_dataset('mano_trans_pred', shape=(num_samples,1,3), dtype='float32', chunks=(64,1,3))
mano_pose_ds = h5_file.create_dataset('mano_pose_pred', shape=(num_samples,1,15,3,3), dtype='float32', chunks=(64,1,15,3,3))
mano_pose_gt_ds = h5_file.create_dataset('mano_pose_gt', shape=(num_samples,1,45), dtype='float32', chunks=(64,1,45))
mano_betas_ds = h5_file.create_dataset('mano_betas_pred', shape=(num_samples,1,10), dtype='float32', chunks=(64,1,10))
mano_side_ds = h5_file.create_dataset('mano_side_gt', shape=(num_samples,1), dtype='int', chunks=(64,1))
mano_xyz_root_ds = h5_file.create_dataset('mano_xyz_root_gt', shape=(num_samples,1,3), dtype='float32', chunks=(64,1,3))
obj_dets_ds = h5_file.create_dataset('obj_dets', shape=(num_samples,1,10), dtype='float32', chunks=(64,1,10))
hand_dets_ds = h5_file.create_dataset('hand_dets', shape=(num_samples,1,10), dtype='float32', chunks=(64,1,10))
import cv2
import numpy as np

# ④ データ書き込みループ
# print(f"dataset[-100]['rgb_path'] : {dataset[-100]['rgb_path']}")
for idx in tqdm(range(num_samples)):
    sample = dataset[idx]

    # print(f"sample['rgb_path'] : {sample['rgb_path']}")

    rgb_data = np.asarray(cv2.imread(sample['rgb_path']))             # (480,640,3)
    # cv2.imwrite(f"/large/naru/HOGraspNet/{idx}.png", rgb_data)
    # print("save!!!!!!")
    taxonomy_id = sample['taxonomy_id']          # int
    object_id = sample['object_id']  


    # mano_pred = sample['mano_hamer']['pred_vertices']
    with open(sample['mano_path'], 'rb') as d:
        mano_pred = pickle.load(d)
    # print(f"mano_pred : {mano_pred.keys()}")
    # print(f"mano_pred : {mano_pred['pred_mano_params']}")

    mano_params_pred = mano_pred['pred_mano_params']
    mano_pose_pred = mano_params_pred["hand_pose"].cpu().numpy()
    mano_vertices_pred = mano_pred["pred_vertices"]
    if isinstance(mano_vertices_pred, (np.ndarray, torch.Tensor)):
        mano_vertices_pred = mano_vertices_pred
    else:
        mano_vertices_pred = np.void(pickle.dumps(mano_vertices_pred))


    with open(sample['bbox_path'], 'rb') as d:
        bbox = pickle.load(d)

    if bbox['hand_dets'] is None or bbox['obj_dets'] is None:
        hand_dets = np.zeros((1, 10), dtype=np.float32)
        obj_dets = np.zeros((1, 10), dtype=np.float32)  
    else:
        hand_dets, obj_dets = select_nearest_hand_obj_pair(bbox)

    rgb_ds[idx] = rgb_data
    taxonomy_ds[idx] = taxonomy_id
    object_ds[idx] = object_id
    mano_ds[idx] = mano_vertices_pred
    obj_dets_ds[idx] = obj_dets
    hand_dets_ds[idx] = hand_dets
    mano_trans_ds[idx] = sample['mano_trans']
    mano_pose_ds[idx] = mano_pose_pred
    mano_pose_gt_ds[idx] = sample['mano_pose']

    mano_betas_ds[idx] = sample['mano_betas']
    mano_side_ds[idx] = 1 if sample['mano_side'] == 'right' else 0
    mano_xyz_root_ds[idx] = sample['mano_xyz_root']
    del rgb_data, taxonomy_id, object_id, mano_pred, bbox, hand_dets, obj_dets
    if idx % 10000 == 0:
        gc.collect()

h5_file.close()
# batch_size = 5000
# current_batch = 0
# current_idx = 0

# for idx in tqdm(range(num_samples)):
#     sample = dataset[idx]
#     if sample['bbox']['hand_dets'] is None or sample['bbox']['obj_dets'] is None:
#         continue

#     if current_idx % batch_size == 0:
#         if current_idx != 0:
#             h5_file.close()
#             current_batch += 1
#         h5_path = f"/large/naru/HOGraspNet/data/h5/train_part{current_batch}.h5"
#         h5_file = h5py.File(h5_path, 'w')
#         rgb_ds = h5_file.create_dataset('rgb', shape=(batch_size, 480, 640, 3), dtype='uint8')
#         taxonomy_ds = h5_file.create_dataset('taxonomy_id', shape=(batch_size,), dtype='int')
#         object_ds = h5_file.create_dataset('object_id', shape=(batch_size,), dtype='int')
#         mano_ds = h5_file.create_dataset('mano_hamer', shape=(batch_size,1,778,3), dtype='float32')
#         obj_dets_ds = h5_file.create_dataset('obj_dets', shape=(batch_size,1,10), dtype='float32')
#         hand_dets_ds = h5_file.create_dataset('hand_dets', shape=(batch_size,1,10), dtype='float32')

#         batch_sample_idx = 0

#     rgb_data = sample['rgb_data']
#     taxonomy_id = sample['taxonomy_id']
#     object_id = sample['object_id']
#     mano_pred = sample['mano_hamer']['pred_vertices']
#     if isinstance(mano_pred, (np.ndarray, torch.Tensor)):
#         mano_hamer = mano_pred
#     else:
#         mano_hamer = np.void(pickle.dumps(mano_pred))
#     bbox = sample['bbox']
#     hand_dets, obj_dets = select_nearest_hand_obj_pair(bbox)

#     rgb_ds[batch_sample_idx] = rgb_data
#     taxonomy_ds[batch_sample_idx] = taxonomy_id
#     object_ds[batch_sample_idx] = object_id
#     mano_ds[batch_sample_idx] = mano_hamer
#     obj_dets_ds[batch_sample_idx] = obj_dets
#     hand_dets_ds[batch_sample_idx] = hand_dets

#     batch_sample_idx += 1
#     current_idx += 1

# h5_file.close()
print(f"Saved all to {h5_path}")