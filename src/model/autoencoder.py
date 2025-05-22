import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, contact=True, base_channels=8, latent_dim=16):
        super(Encoder, self).__init__()
        input_dim = 4 if contact else 3
        print(f"input_dim: {input_dim}")

        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(base_channels * 8),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(base_channels * 8, base_channels * 16, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(base_channels * 16),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(base_channels * 16, base_channels * 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(base_channels * 32),
            nn.ReLU()
        )

        self.final_size = 25  # 元のコードでは最後に25長さまで縮む前提
        self.fc = nn.Sequential(
            nn.Linear(base_channels * 32 * self.final_size, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        latent = self.fc(x)
        return latent

class Decoder(nn.Module):
    def __init__(self, contact=True, base_channels=8, latent_dim=16):
        super(Decoder, self).__init__()
        output_dim = 4 if contact else 3
        self.base_channels = base_channels
        self.fc = nn.Linear(latent_dim, base_channels * 32 * 25)

        self.conv1 = nn.Conv1d(base_channels * 32, base_channels * 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(base_channels * 16, base_channels * 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(base_channels, output_dim, kernel_size=3, padding=1)
        
    def forward(self, latent):
        x = self.fc(latent).view(-1, self.base_channels * 32, 25)
        x = F.interpolate(x, size=50, mode='nearest')
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, size=100, mode='nearest')
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, size=200, mode='nearest')
        x = F.relu(self.conv3(x))
        x = F.interpolate(x, size=400, mode='nearest')
        x = F.relu(self.conv4(x))
        x = F.interpolate(x, size=778, mode='nearest')
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        return x

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=8, num_classes=33):
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
    def __init__(self, contact=True, base_channels=64, latent_dim=512, mlp_hidden_dim=128, num_classes=33):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(contact=contact, base_channels=base_channels, latent_dim=latent_dim)
        self.decoder = Decoder(contact=contact, base_channels=base_channels, latent_dim=latent_dim)
        self.classifier = MLPClassifier(input_dim=latent_dim, hidden_dim=mlp_hidden_dim, num_classes=num_classes)
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        classification = self.classifier(latent)
        return reconstructed, classification

if __name__ == "__main__":
    model = Autoencoder(contact=True)
    # 入力サイズ: (batch_size, 4, 778)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    x = torch.randn(2, 4, 778)  # 例: バッチサイズ2
    reconstructed, classfication = model(x)
    print("出力サイズ:", reconstructed.shape, classfication.shape)  # 期待される出力: (2, 4, 778)
