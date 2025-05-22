import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append("/large/naru/HOGraspNet/src")
from dataset.dataset_mano import HOGDataset
from model.image_aux import TaxonomyClassifier
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

def main(cfg):
    wandb.init(project="taxonomy_classification", 
               config=OmegaConf.to_container(cfg, resolve=True),
               name=f"{cfg.model.input_type}_{cfg.model.mano_type}")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = HOGDataset(split="train", db_path=cfg.dataset.db_path, transform=transform, input_type=cfg.model.input_type, mano_type=cfg.model.mano_type, preload=cfg.dataset.preload, noise_level=cfg.train.noise_level)
    val_dataset = HOGDataset(split="test", db_path=cfg.dataset.db_path, transform=transform, input_type=cfg.model.input_type, mano_type=cfg.model.mano_type, preload=cfg.dataset.preload, noise_level=cfg.train.noise_level)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.dataset.num_workers, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=False)

    # --- モデルとResNet50セットアップ ---
    model = TaxonomyClassifier(
        cfg.model.mano_input_dim, 
        cfg.model.img_input_dim, 
        cfg.model.hidden_dim, 
        cfg.model.num_classes, 
        cfg.model.input_type,
        cfg.model.mano_type
    )
    parameters = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {parameters}")

    if cfg.model.type == "image_net":
        img_enc = nn.Sequential(*(list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-1]))
    elif cfg.model.type == "simhand":
        resnet = resnet50()
        resnet.load_state_dict(torch.load("/large/naru/exp_hamer/hamer/resnet50_simhand.pth"))
        img_enc = nn.Sequential(*(list(resnet.children())[:-1]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    img_enc = img_enc.to(device)

    if cfg.model.img_input_dim != 2048:
        img_projector = nn.Linear(2048, cfg.model.img_input_dim).to(device)
    else:
        img_projector = nn.Identity()

    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=cfg.train.patience)

    weight = (torch.tensor(cfg.train.class_weights) / sum(cfg.train.class_weights)) * cfg.model.num_classes
    criterion_cls = nn.CrossEntropyLoss(weight=weight.to(device))
    criterion_aux = nn.MSELoss()
    lambda_aux = cfg.train.lambda_aux  # e.g., 0.1

    best_val_acc = 0
    for epoch in range(cfg.train.num_epochs):
        model.train()
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg.train.num_epochs} [Train]")
        train_loss_total, train_correct, train_total = 0, 0, 0

        for batch in train_pbar:
            for key, value in batch.items():
                batch[key] = value.to(device)
            img_enc.eval()
            with torch.no_grad():
                img_feat = img_enc(batch["img_feat"]).view(batch["img_feat"].size(0), -1)
                img_feat = img_projector(img_feat)
            batch["img_feat"] = img_feat

            output = model(batch)
            logits = output['classification']
            labels = batch["taxonomy_id"]

            loss_cls = criterion_cls(logits, labels)
            total_loss = loss_cls

            # 補助タスク（画像からpose予測）
            # if 'img_pose_regression' in output and output['img_pose_regression'] is not None:
            if False:
                target_pose = batch["mano_pose"].squeeze(1)
                loss_aux = criterion_aux(output['img_pose_regression'], target_pose)
                total_loss += lambda_aux * loss_aux
            else:
                loss_aux = torch.tensor(0.0)

            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean().item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss_total += total_loss.item() * labels.size(0)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            train_pbar.set_postfix({
                'loss_cls': f"{loss_cls.item():.4f}",
                'loss_aux': f"{loss_aux.item():.4f}",
                'acc': f"{acc:.4f}"
            })
            wandb.log({
                'epoch': epoch + 1,
                'batch_train_loss_cls': loss_cls.item(),
                'batch_train_loss_aux': loss_aux.item(),
                'batch_train_acc': acc
            })

        avg_train_loss = train_loss_total / train_total
        avg_train_acc = train_correct / train_total

        # --- 検証ループ ---
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix

        model.eval()
        val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{cfg.train.num_epochs} [Val]")
        val_loss_total, val_correct, val_correct_top2, val_total = 0, 0, 0, 0
        all_preds, all_labels = [], []
        zero_acc_batches_top1 = []
        zero_acc_batches_top2 = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                for key, value in batch.items():
                    batch[key] = value.to(device)
                img_feat = img_enc(batch["img_feat"]).view(batch["img_feat"].size(0), -1)
                img_feat = img_projector(img_feat)
                batch["img_feat"] = img_feat

                output = model(batch)
                logits = output['classification']
                labels = batch["taxonomy_id"]

                loss_cls = criterion_cls(logits, labels)
                total_loss = loss_cls

                if 'img_pose_regression' in output and output['img_pose_regression'] is not None:
                    target_pose = batch["mano_pose"].squeeze(1)
                    loss_aux = criterion_aux(output['img_pose_regression'], target_pose)
                    total_loss += lambda_aux * loss_aux

                preds = torch.argmax(logits, dim=1)
                correct_top1 = (preds == labels).float()

                topk = cfg.train.topk
                _, topk_preds = torch.topk(logits, topk, dim=1)
                labels_expanded = labels.view(-1, 1).expand_as(topk_preds)
                correct_top2 = (topk_preds == labels_expanded).any(dim=1).float()

                acc_top1 = correct_top1.mean().item()
                acc_top2 = correct_top2.mean().item()

                if acc_top1 == 0.0:
                    zero_acc_batches_top1.append(batch_idx)
                    print(f"Zero-acc batch {batch_idx}: labels={labels.cpu().tolist()}, preds={preds.cpu().tolist()}")
                if acc_top2 == 0.0:
                    zero_acc_batches_top2.append(batch_idx)
                    print(f"Zero-acc-top2 batch {batch_idx}: labels={labels.cpu().tolist()}, preds={topk_preds.cpu().tolist()}")

                val_loss_total += total_loss.item() * labels.size(0)
                val_correct += correct_top1.sum().item()
                val_correct_top2 += correct_top2.sum().item()
                val_total += labels.size(0)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

                val_pbar.set_postfix({
                    'batch_loss': f"{total_loss.item():.4f}",
                    'batch_acc_top1': f"{acc_top1:.4f}",
                    'batch_acc_top2': f"{acc_top2:.4f}"
                })

        avg_val_loss = val_loss_total / val_total
        avg_val_acc_top1 = val_correct / val_total
        avg_val_acc_top2 = val_correct_top2 / val_total
        scheduler.step(avg_val_loss)

        print(f"Batches with zero top-1 accuracy: {zero_acc_batches_top1}")
        print(f"Batches with zero top-2 accuracy: {zero_acc_batches_top2}")

        # ---- confusion matrix ----
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(cfg.model.num_classes)))

        # Normalize row-wise
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        cm_normalized = cm.astype('float') / row_sums

        # Save plot
        os.makedirs(cfg.save_path.confusion_matrix, exist_ok=True)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Normalized Confusion Matrix")
        plt.colorbar()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f"{cfg.save_path.confusion_matrix}/{epoch}.png")
        print(f"Saved normalized confusion matrix heatmap as confusion_matrix_normalized_{epoch}.png")

        # Per-class accuracy
        correct_per_class = np.diag(cm)
        total_per_class = cm.sum(axis=1)
        class_accuracy = {}
        for i in range(cfg.model.num_classes):
            total = total_per_class[i]
            acc = (correct_per_class[i] / total) if total > 0 else 0.0
            class_accuracy[f"class_{i}_accuracy"] = acc

        # wandb log
        wandb.log({
            'epoch': epoch + 1,
            'avg_train_loss': avg_train_loss,
            'avg_train_acc': avg_train_acc,
            'avg_val_loss': avg_val_loss,
            'avg_val_acc_top1': avg_val_acc_top1,
            'avg_val_acc_top2': avg_val_acc_top2,
            'lr': optimizer.param_groups[0]['lr'],
            **class_accuracy
        })

        # Checkpoint save
        if avg_val_acc_top1 > best_val_acc:
            best_val_acc = avg_val_acc_top1
            save_path = os.path.join(cfg.save_path.checkpoint, f"best_model_epoch_{epoch + 1}.pth")
            os.makedirs(cfg.save_path.checkpoint, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc_top1': best_val_acc,
            }, save_path)
            print(f"Saved new best checkpoint at {save_path}")

if __name__ == "__main__":
    cfg = OmegaConf.load("src/cfg/cfg_mano.yaml")
    main(cfg)