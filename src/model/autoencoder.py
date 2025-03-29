import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 入力: (batch, 4, 778)
        self.conv1 = nn.Conv1d(4, 32, kernel_size=3, padding=1)  # -> (batch, 32, 778)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1, stride=2)  # -> (batch, 64, 389)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=2)  # -> (batch, 128, 195)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=2)  # -> (batch, 256, 98)
        self.conv5 = nn.Conv1d(256, 512, kernel_size=3, padding=1, stride=2)  # -> (batch, 512, 49)
        self.conv6 = nn.Conv1d(512, 1024, kernel_size=3, padding=1, stride=2)  # -> (batch, 1024, 25)

        self.fc = nn.Linear(1024 * 25, 256)  # 潜在表現のサイズを 256 に拡張
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.view(x.size(0), -1)
        latent = self.fc(x)
        return latent

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(256, 1024 * 25)

        self.conv1 = nn.Conv1d(1024, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(32, 4, kernel_size=3, padding=1)
        
    def forward(self, latent):
        x = self.fc(latent).view(-1, 1024, 25)
        x = F.interpolate(x, size=50, mode='nearest')  # 50 に固定
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, size=100, mode='nearest')  # 100 に固定
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, size=200, mode='nearest')  # 200 に固定
        x = F.relu(self.conv3(x))
        x = F.interpolate(x, size=400, mode='nearest')  # 400 に固定
        x = F.relu(self.conv4(x))
        x = F.interpolate(x, size=778, mode='nearest')  # 最後に 778 に調整
        x = F.relu(self.conv5(x))
        x = self.conv6(x)  # -> (batch, 4, 778)
        return x

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_classes=33):
        super(MLPClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),  
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.classifier = MLPClassifier()
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        classification = self.classifier(latent)
        return reconstructed, classification

if __name__ == "__main__":
    model = Autoencoder()
    # 入力サイズ: (batch_size, 4, 778)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    x = torch.randn(2, 4, 778)  # 例: バッチサイズ2
    reconstructed, classfication = model(x)
    print("出力サイズ:", reconstructed.shape, classfication.shape)  # 期待される出力: (2, 4, 778)
