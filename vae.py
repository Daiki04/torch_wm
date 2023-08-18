import torch
import torch.nn as nn
from torch.nn import functional as F

class Encoder(nn.Module):
    """ Encoder"""

    def __init__(self, z_dim=32):
        super().__init__()

        self.z_dim = z_dim # 潜在表現の次元数

        # 畳み込み層
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)

        self.mu = nn.Linear(1024, z_dim) # 平均を表現する線形層
        self.log_var = nn.Linear(1024, z_dim) # 分散の対数値を表現する線形層

    def forward(self, x):
        # x: [batch_size, 3, 64, 64]
        c1 = F.relu(self.conv1(x)) # [batch_size, 3, 64, 64] -> [batch_size, 32, 31, 31]
        c2 = F.relu(self.conv2(c1)) # [batch_size, 32, 31, 31] -> [batch_size, 64, 14, 14]
        c3 = F.relu(self.conv3(c2)) # [batch_size, 64, 14, 14] -> [batch_size, 128, 6, 6]
        c4 = F.relu(self.conv4(c3)) # [batch_size, 128, 6, 6] -> [batch_size, 256, 2, 2]

        d1 = c4.view(-1, 1024) # [batch_size, 256, 2, 2] -> [batch_size, 1024]

        mu = self.mu(d1) # [batch_size, 1024] -> [batch_size, z_dim]
        log_var = self.log_var(d1) # [batch_size, 1024] -> [batch_size, z_dim]
        ep = torch.randn_like(log_var) # [batch_size, z_dim], N(0, 1), 正規分布からサンプリング

        z = mu + ep * torch.exp(log_var / 2) # [batch_size, z_dim], 潜在表現をサンプリング

        return z, mu, log_var
    
class Decoder(nn.Module):
    """Decoder"""

    def __init__(self, z_dim=32):
        super().__init__()

        self.z_dim = z_dim

        self.linear = nn.Linear(z_dim, 1024) # 潜在表現を線形変換する層

        # 転置畳み込み層
        self.conv1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2)
    
    def forward(self, x):
        # x: [batch_size, z_dim]
        d1 = self.linear(x) # [batch_size, z_dim] -> [batch_size, 1024]
        d1 = d1.view(-1, 1024, 1, 1) # [batch_size, 1024] -> [batch_size, 1024, 1, 1]

        ct1 = F.relu(self.conv1(d1)) # [batch_size, 1024, 1, 1] -> [batch_size, 128, 5, 5]
        ct2 = F.relu(self.conv2(ct1)) # [batch_size, 128, 5, 5] -> [batch_size, 64, 13, 13]
        ct3 = F.relu(self.conv3(ct2)) # [batch_size, 64, 13, 13] -> [batch_size, 32, 30, 30]
        ct4 = F.sigmoid(self.conv4(ct3)) # [batch_size, 32, 30, 30] -> [batch_size, 3, 64, 64]

        return ct4
    
class VAE(nn.Module):
    """VAE"""

    def __init__(self, z_dim=32):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(z_dim) # エンコーダ
        self.decoder = Decoder(z_dim) # デコーダ

    def forward(self, x):
        # x: [batch_size, 3, 64, 64]
        z, mu, log_var = self.encoder(x) # [batch_size, 3, 64, 64] -> [batch_size, z_dim], [batch_size, z_dim], [batch_size, z_dim]
        y = self.decoder(z) # [batch_size, z_dim] -> [batch_size, 3, 64, 64]

        return y, mu, log_var
    
    def encode(self, x):
        ### 画像を潜在表現に変換する ###
        # x: [batch_size, 3, 64, 64]
        z, mu, log_var = self.encoder(x) # [batch_size, z_dim], [batch_size, z_dim], [batch_size, z_dim]
        return z, mu, log_var
    
    def decode(self, z):
        ### 潜在表現から画像を生成する ###
        # z: [batch_size, z_dim]
        y = self.decoder(z) # [batch_size, 3, 64, 64]
        return y
    
    def r_loss(self, y_true, y_pred):
        ### 再構成誤差(RMSE)を計算する ###
        # y_true: [batch_size, 3, 64, 64]
        # y_pred: [batch_size, 3, 64, 64]
        # rmse = (y_true - y_pred) ** 2 # [batch_size, 3, 64, 64]
        # rmse = torch.sum(rmse, dim=(1, 2, 3)) # [batch_size]
        # rmse = torch.sqrt(torch.mean(rmse)) # 1つの値にスカラー変換
        rmse = nn.MSELoss(reduction='none')(y_true, y_pred).mean(dim=(1, 2, 3))
        rmse = torch.sqrt(rmse)
        rmse = torch.mean(rmse)
        return rmse
        
    def kl_loss(self, mu, log_var, kl_tolerance):
        ### 潜在表現の分布と事前分布のKLダイバージェンスを計算する ###
        # mu: [batch_size, z_dim]
        # log_var: [batch_size, z_dim]
        kl = -0.5 * torch.sum(1 + log_var - mu**2 - torch.exp(log_var), dim=1) # [batch_size]
        kl = torch.max(kl, kl.new([self.z_dim * kl_tolerance])) # [batch_size]
        kl = torch.mean(kl) # 1つの値にスカラー変換
        return kl
    
    def loss(self, y_true, y_pred, mu, log_var, kl_tolerance=0.5):
        ### VAEの損失関数を計算する: r_loss + kl_loss ###
        # y_true: [batch_size, 3, 64, 64]
        # y_pred: [batch_size, 3, 64, 64]
        # mu: [batch_size, z_dim]
        # log_var: [batch_size, z_dim]
        r_loss = self.r_loss(y_true, y_pred) # [batch_size]
        kl_loss = self.kl_loss(mu, log_var, kl_tolerance) # [batch_size]
        loss = r_loss + kl_loss # スカラー値
        return loss, r_loss, kl_loss