import torch
import h5py
import sys
sys.path.append('/large/naru/HOGraspNet')
from src.dataset.dataloader import HOGDataLoader
from src.model.autoencoder import Autoencoder
from tqdm import tqdm
checkpoint = torch.load('/large/naru/HOGraspNet/checkpoints/model_epoch_200.pth')
model = Autoencoder()  # モデルのインスタンスを作成
model.load_state_dict(checkpoint['model_state_dict'])  # 辞書からモデルの重みをロード
model.eval() 
model.to('cuda')
val_loader = HOGDataLoader('val', 'processed_data1', 
                            batch_size=1, 
                            num_workers=1, 
                            shuffle=False, 
                            contact_bin=False)
total = 0
correct = 0
for i, (data, label) in tqdm(enumerate(val_loader), total=len(val_loader)):
    data, label = data.to('cuda').transpose(1, 2), label.to('cuda')
    reconstructed, classification = model(data)
    _, predicted = torch.max(classification.data, 1)
    labels = label.argmax(dim=1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    print(classification)
    if predicted != labels:
        print(classification)
        print(labels)
    if i == 20:
        break

print(f"Accuracy: {correct / total}")

