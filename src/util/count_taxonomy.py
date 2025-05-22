import sys
sys.path.append("/large/naru/HOGraspNet/src")
from model.classify import TaxonomyClassifier
from dataset.dataset_mano import HOGDataset
from torch.utils.data import DataLoader
import cv2
from omegaconf import OmegaConf
from tqdm import tqdm
from collections import Counter
import os
cfg = OmegaConf.load("src/cfg/cfg_mano.yaml")
val_dataset = HOGDataset(split="train", db_path=cfg.dataset.db_path, input_type=cfg.model.input_type, mano_type=cfg.model.mano_type, preload=cfg.dataset.preload, noise_level=cfg.train.noise_level)
val_dataloader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=False)
os.makedirs("rgb", exist_ok=True)


taxonomy_counter = Counter()

for i, batch in tqdm(enumerate(val_dataloader)):
    taxonomy_ids = batch["taxonomy_id"]

    # taxonomy_ids が tensor の場合、list に変換
    if hasattr(taxonomy_ids, 'tolist'):
        taxonomy_ids = taxonomy_ids.tolist()

    # Counter を更新
    taxonomy_counter.update(taxonomy_ids)

# 出現頻度を表示
for tax_id, count in taxonomy_counter.most_common():
    print(f"Taxonomy ID {tax_id}: {count} times")

total_count = sum(taxonomy_counter.values())

# クラス数を取得（最大ID+1で仮定）
num_classes = max(taxonomy_counter.keys()) + 1

# 各クラスの頻度を0で初期化
class_counts = [0] * num_classes

for tax_id, count in taxonomy_counter.items():
    class_counts[tax_id] = count

# 重み計算：出現頻度が少ないほど大きな重みを持たせる（例：逆頻度）
weights = [total_count / c if c > 0 else 0.0 for c in class_counts]

print(weights)
print(len(weights))