import torch
import h5py
import sys
sys.path.append('/large/naru/HOGraspNet')
from src.dataset.dataloader import HOGDataLoader
from src.model.autoencoder import Autoencoder
from tqdm import tqdm
from torch.nn import functional as F
checkpoint = torch.load('/large/naru/HOGraspNet/checkpoints_all_contact2/model_epoch_200.pth')
contact = False

model = Autoencoder(contact=contact)  # モデルのインスタンスを作成
model.load_state_dict(checkpoint['model_state_dict'])  # 辞書からモデルの重みをロード
model.eval() 
model.to('cuda')
val_loader = HOGDataLoader('test', 'processed_all_data', 
                            batch_size=1, 
                            num_workers=1, 
                            shuffle=False, 
                            contact_bin=False,
                            contact=contact)
print(len(val_loader))
total = 0
correct = 0
for i, (data, label) in tqdm(enumerate(val_loader), total=len(val_loader)):
    data, label = data.to('cuda').transpose(1, 2), label.to('cuda')
    reconstructed, classification = model(data)
    probs = F.softmax(classification, dim=-1)
    # print(f"probs: {probs[0]}")
    _, predicted = torch.max(probs.data, 1)
    labels = label.argmax(dim=1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    # print(classification)
    # if predicted != labels:
        # print(classification)
        # print(labels)
    # if i == 20:
    #     break

print(f"Accuracy: {correct / total}")

