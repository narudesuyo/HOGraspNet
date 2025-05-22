import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))

class TaxonomyClassifier(nn.Module):
    def __init__(self, 
                 mano_input_dim=45, 
                 img_input_dim=512, 
                 hidden_dim=512,  
                 num_classes=33, 
                 input_type=["mano", "image"],
                 mano_type="gt",
                 dropout_rate=0.5):
        super(TaxonomyClassifier, self).__init__()
        
        self.use_mano = "mano" in input_type
        self.use_image = "image" in input_type
        self.mano_type = mano_type
        self.dropout_rate = dropout_rate

        def make_branch(input_dim):
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                ResidualBlock(hidden_dim),
                nn.Dropout(self.dropout_rate),
            )
        
        if self.use_mano:
            self.mano_branch = make_branch(mano_input_dim)
        
        if self.use_image:
            self.img_branch = make_branch(img_input_dim)
            # 補助タスク用回帰ヘッド（画像からpose予測）
            self.img_pose_regressor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, mano_input_dim)
            )
        
        combined_dim = (hidden_dim if self.use_mano else 0) + (hidden_dim if self.use_image else 0)
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            ResidualBlock(hidden_dim),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def update_dropout(self, rate):
        """Dropout rate をスケジュール調整する用関数"""
        def set_dropout(m):
            if isinstance(m, nn.Dropout):
                m.p = rate
        self.apply(set_dropout)

    def forward(self, input_dict):
        features = []
        img_pose_regression_output = None

        if self.use_mano:
            mano_feat = input_dict.get('mano_pose', None)
            if mano_feat is None:
                raise ValueError("Model is set to use MANO input but 'mano_pose' is missing in input_dict")
            if len(mano_feat.shape) == 3:
                mano_feat = mano_feat.squeeze(1)
            mano_out = self.mano_branch(mano_feat)
            features.append(mano_out)

        if self.use_image:
            img_feat = input_dict.get('img_feat', None)
            if img_feat is None:
                raise ValueError("Model is set to use Image input but 'img_feat' is missing in input_dict")
            img_out = self.img_branch(img_feat)
            features.append(img_out)
            img_pose_regression_output = self.img_pose_regressor(img_out)  # 補助タスク出力
        
        combined = torch.cat(features, dim=1) if len(features) > 1 else features[0]
        class_output = self.classifier(combined)

        return {
            'classification': class_output,
            'img_pose_regression': img_pose_regression_output  # 補助タスク用
        }