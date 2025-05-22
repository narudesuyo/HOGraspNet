import sys
sys.path.append("/large/naru/HOGraspNet/src")
from model.classify import TaxonomyClassifier
from dataset.dataset_mano import HOGDataset
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
from tqdm import tqdm

def evaluate(cfg, ckpt_path):
    # --- データ前処理 ---
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = HOGDataset(split="test", db_path=cfg.dataset.db_path, transform=transform, input_type=cfg.model.input_type, mano_type=cfg.model.mano_type, preload=cfg.dataset.preload, noise_level=cfg.train.noise_level)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=False)

    # --- モデルとResNet50セットアップ ---
    model = TaxonomyClassifier(
        cfg.model.mano_input_dim, 
        cfg.model.img_input_dim, 
        cfg.model.hidden_dim, 
        cfg.model.num_classes, 
        cfg.model.input_type,
        cfg.model.mano_type
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 画像エンコーダ（ResNet）
    if cfg.model.type == "image_net":
        img_enc = nn.Sequential(*(list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-1]))
    elif cfg.model.type == "simhand":
        resnet = resnet50()
        resnet.load_state_dict(torch.load("/large/naru/exp_hamer/hamer/resnet50_simhand.pth"))
        img_enc = nn.Sequential(*(list(resnet.children())[:-1]))
    img_enc = img_enc.to(device)
    img_enc.eval()

    if cfg.model.img_input_dim != 2048:
        img_projector = nn.Linear(2048, cfg.model.img_input_dim).to(device)
    else:
        img_projector = nn.Identity()

    # --- checkpoint ロード ---
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 損失関数（評価用）
    criterion = nn.CrossEntropyLoss()

    # --- 評価ループ ---
    all_preds = []
    all_labels = []
    total_loss, correct_top1, correct_top2, total = 0, 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            for key, value in batch.items():
                batch[key] = value.to(device)
            img_feat = img_enc(batch["img_feat"])
            img_feat = img_feat.view(img_feat.size(0), -1)
            img_feat = img_projector(img_feat)
            batch["img_feat"] = img_feat

            output = model(batch)
            labels = batch["taxonomy_id"]

            loss = criterion(output, labels)

            preds_top1 = torch.argmax(output, dim=1)
            correct_top1_batch = (preds_top1 == labels).float()

            topk = cfg.train.topk
            _, topk_preds = torch.topk(output, topk, dim=1)
            labels_expanded = labels.view(-1, 1).expand_as(topk_preds)
            correct_top2_batch = (topk_preds == labels_expanded).any(dim=1).float()

            total_loss += loss.item() * labels.size(0)
            correct_top1 += correct_top1_batch.sum().item()
            correct_top2 += correct_top2_batch.sum().item()
            total += labels.size(0)

            all_preds.extend(preds_top1.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total
    acc_top1 = correct_top1 / total
    acc_top2 = correct_top2 / total

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Top-1 Accuracy: {acc_top1:.4f}")
    print(f"Top-{topk} Accuracy: {acc_top2:.4f}")

    # 混同行列
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(cfg.model.num_classes)))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_normalized = cm.astype('float') / row_sums

    os.makedirs(cfg.save_path.confusion_matrix, exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Normalized Confusion Matrix (Test Set)")
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{cfg.save_path.confusion_matrix}/test_confusion_matrix.png")
    print("Saved confusion matrix plot as test_confusion_matrix.png")

if __name__ == '__main__':
    cfg = OmegaConf.load('src/cfg/cfg_mano.yaml')
    ckpt_path = '/large/naru/HOGraspNet/checkpoint/ckpt/best_model_epoch_8.pth'  # ← ここを実際のckptパスに置き換える
    evaluate(cfg, ckpt_path)