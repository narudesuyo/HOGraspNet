import sys
sys.path.append('/large/naru/HOGraspNet/src')
from model.autoencoder import Autoencoder
from dataset.dataset import HOGDataset
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

os.makedirs('checkpoints', exist_ok=True)


@profile
def train_autoencoder(model: Autoencoder,
                     train_loader: DataLoader,
                     val_loader: DataLoader,
                     num_epochs: int = 100,
                     learning_rate: float = 1e-3,
                     device: str = 'cuda',
                     lambda_recon: float = 1.0,
                     lambda_class: float = 1.0,
                     val_interval: int = 10):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    model = model.to(device)
    print(next(model.parameters()).device)

    criterion1 = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model_dir = 'checkpoints_no_contact'
    os.makedirs(model_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for combined_data, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            combined_data, labels = combined_data.transpose(1, 2).to(device), labels.to(device)

            optimizer.zero_grad()
            
            # forward pass
            reconstructed, classfication = model(combined_data)
            loss1 = criterion1(reconstructed, combined_data)
            loss2 = criterion2(classfication, labels)
            loss = lambda_recon * loss1 + lambda_class * loss2
            
            # backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        

        if epoch % val_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                correct = 0
                total = 0
                for combined_data, labels in val_loader:
                    combined_data, labels = combined_data.transpose(1, 2).to(device), labels.to(device)
                    reconstructed, classfication = model(combined_data) 
                    loss1 = criterion1(reconstructed, combined_data)
                    loss2 = criterion2(classfication, labels)
                    loss = lambda_recon * loss1 + lambda_class * loss2
                    val_loss += loss.item()

                    _, predicted = torch.max(classfication.data, 1)
                    labels = labels.argmax(dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
            avg_val_loss = val_loss / len(val_loader)

        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "epoch": epoch,
            "accuracy": correct / total
        })       
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], '
                   f'Train Loss: {avg_train_loss:.4f}, '
                   f'Val Loss: {avg_val_loss:.4f}, '
                   f'Accuracy: {correct / total:.4f} ')
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
    contact = False
    wandb.init(project="HOG_autoencoder")
    print(torch.cuda.is_available())
    mp.set_start_method('spawn')
    model = Autoencoder(contact=contact)
    print("model parameters:", sum(p.numel() for p in model.parameters()))
    setup = 's2'
    db_path = "processed_data1"

    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    train_config = config['train']
    data_config = config['data']

    train_loader = HOGDataLoader('train', db_path, 
                                 batch_size=data_config['batch_size'], 
                                 num_workers=data_config['num_workers'], 
                                 shuffle=data_config['shuffle'], 
                                 contact_bin=data_config['contact_bin'],
                                 contact=contact)
    print(f"train_loader: {len(train_loader)}")

    val_loader = HOGDataLoader('val', db_path, 
                               batch_size=data_config['batch_size'], 
                               num_workers=data_config['num_workers'], 
                               shuffle=data_config['shuffle'], 
                               contact_bin=data_config['contact_bin'],
                               contact=contact)
    print(f"val_loader: {len(val_loader)}")

    train_autoencoder(model, train_loader, val_loader,
                      num_epochs=train_config['num_epochs'],
                      learning_rate=train_config['learning_rate'],
                      lambda_recon=train_config['lambda_recon'],
                      lambda_class=train_config['lambda_class'],
                      val_interval=train_config['val_interval'])

    wandb.finish()
