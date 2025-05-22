import sys
sys.path.append('/large/naru/HOGraspNet/src')
from model.autoencoder import Autoencoder
from dataset.dataset import HOGDatasetAll
from dataset.dataloader import HOGDataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import wandb
import os
import torch.multiprocessing as mp
import smplx
from util.calc_mano_mesh import calc_mano_mesh
from memory_profiler import profile
import json

@profile
def train_autoencoder(model: Autoencoder,
                     train_loader: DataLoader,
                     val_loader: DataLoader,
                     num_epochs: int = 100,
                     learning_rate: float = 1e-3,
                     device: str = 'cuda',
                     lambda_recon: float = 1.0,
                     lambda_class: float = 1.0,
                     val_interval: int = 10,
                     model_dir: str = 'checkpoints_all_contact'):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    model = model.to(device)
    print(next(model.parameters()).device)

    criterion1 = nn.MSELoss()
    weight = [1.4620008666110853, 1.3899496698161533, 2.497857982864679, 0.986753463558589, 0.35965929309367134, 0.0, 1.0211225117851943, 0.0, 0.5448983420678146, 1.5146766286303832, 0.8609309585453222, 0.3019091220481428, 0.8215668586708311, 0.28790403420536165, 0.0, 0.2330149885709364, 0.8919492877162906, 0.4256953230254669, 0.5261131808064449, 0.6426562794102376, 0.0, 1.0650522565560678, 2.740577270252514, 0.8799720855905583, 1.9561802640169392, 4.501190546899584, 1.9578992273420859, 0.741707496909219, 0.9241349318603458, 1.0589778140281816, 1.9011001029993975, 0.0, 0.5045492121184996]
    weight = torch.tensor(weight).to(device)
    criterion2 = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)


    os.makedirs(model_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_class_loss = 0
        correct = 0
        total = 0
        
        for combined_data, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            combined_data, labels = combined_data.transpose(1, 2).to(device).float().squeeze(2).permute(0, 2, 1), labels.to(device)

            optimizer.zero_grad()
            
            # forward pass
            reconstructed, classfication = model(combined_data)
            loss1 = criterion1(reconstructed, combined_data)
            loss2 = criterion2(classfication, labels)
            loss = lambda_recon * loss1 + lambda_class * loss2
            
            # backward pass
            loss.backward()
            optimizer.step()

            train_recon_loss += lambda_recon * loss1.item()
            train_class_loss += lambda_class * loss2.item()
            train_loss += loss.item()

            _, predicted = torch.max(classfication.data, 1)
            # labels = labels.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_recon_loss = train_recon_loss / len(train_loader)
        avg_train_class_loss = train_class_loss / len(train_loader)
        train_acc = correct / total
        

        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_class_loss = 0
        with torch.no_grad():
            correct = 0
            total = 0
            for combined_data, labels in tqdm(val_loader, desc="Evaluating", leave=False):
                combined_data, labels = combined_data.transpose(1, 2).to(device).float().squeeze(2).permute(0, 2, 1), labels.to(device)
                reconstructed, classfication = model(combined_data) 
                loss1 = criterion1(reconstructed, combined_data)
                loss2 = criterion2(classfication, labels)
                loss = lambda_recon * loss1 + lambda_class * loss2

                val_recon_loss += lambda_recon * loss1.item()
                val_class_loss += lambda_class * loss2.item()
                val_loss += loss.item()

                _, predicted = torch.max(classfication.data, 1)
                # labels = labels.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = correct / total
        avg_val_loss = val_loss / len(val_loader)
        avg_val_recon_loss = val_recon_loss / len(val_loader)
        avg_val_class_loss = val_class_loss / len(val_loader)
        scheduler.step(avg_val_loss)


        wandb.log({
            "train_loss": avg_train_loss,
            "train_recon_loss": avg_train_recon_loss,
            "train_class_loss": avg_train_class_loss,
            "val_loss": avg_val_loss,
            "val_recon_loss": avg_val_recon_loss,
            "val_class_loss": avg_val_class_loss,
            "epoch": epoch,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]['lr']
        })       
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], '
                   f'Train Loss: {avg_train_loss:.4f}, '
                   f'Train Recon Loss: {avg_train_recon_loss:.4f}, '
                   f'Train Class Loss: {avg_train_class_loss:.4f}, '
                   f'Val Loss: {avg_val_loss:.4f}, '
                   f'Val Recon Loss: {avg_val_recon_loss:.4f}, '
                   f'Val Class Loss: {avg_val_class_loss:.4f}, '
                   f'Train Acc: {train_acc:.4f}, '
                   f'Val Acc: {val_acc:.4f}, '
                   f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        if (epoch + 1) % 5 == 0:  # 5エポックごとに保存
            model_path = os.path.join(model_dir, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, model_path)
            logger.info(f'Model saved at {model_path}')

if __name__ == '__main__':


    with open('config/config.json', 'r') as f:
        config = json.load(f)
    wandb.init(project="HOG_autoencoder")
    wandb.config.update(config)

    print(torch.cuda.is_available())
    mp.set_start_method('spawn')
    model = Autoencoder(contact=config['train']['contact'], base_channels=config['model']['base_channels'], latent_dim=config['model']['latent_dim'], mlp_hidden_dim=config['model']['mlp_hidden_dim'], num_classes=config['model']['num_classes'])
    print("model parameters:", sum(p.numel() for p in model.parameters()))
    setup = 's3'
    db_path = "data/h5_s3_fil"
    train_config = config['train']
    data_config = config['data']

    train_loader = HOGDataLoader('train', db_path, 
                                 batch_size=data_config['batch_size'], 
                                 num_workers=data_config['num_workers'], 
                                 shuffle=data_config['shuffle'], 
                                 contact_bin=data_config['contact_bin'],
                                 contact=train_config['contact'],
                                 max_angle=data_config['max_angle'],
                                 vertices_aug=data_config['vertices_aug'],
                                 noise_level=data_config['noise_level'])
    print(f"train_loader: {len(train_loader)}")

    val_loader = HOGDataLoader('test', db_path, 
                               batch_size=data_config['batch_size'], 
                               num_workers=data_config['num_workers'], 
                               shuffle=data_config['shuffle'], 
                               contact_bin=data_config['contact_bin'],
                               contact=train_config['contact'])
    print(f"val_loader: {len(val_loader)}")
    model_dir = 'checkpoints_s3'
    train_autoencoder(model, train_loader, val_loader,
                      num_epochs=train_config['num_epochs'],
                      learning_rate=train_config['learning_rate'],
                      lambda_recon=train_config['lambda_recon'],
                      lambda_class=train_config['lambda_class'],
                      val_interval=train_config['val_interval'],
                      model_dir=model_dir)
    config_save_path = os.path.join(model_dir, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f)

    wandb.finish()
