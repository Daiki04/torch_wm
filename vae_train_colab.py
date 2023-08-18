import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from rollout_colab import CarRacing_rollouts
from vae import VAE

def main():
    num_epoch = 3 # エポック数
    batch_size = 2 # バッチサイズ
    lr = 0.0001 # 学習率

    seq_len = 300 # シーケンスの長さ
    num_rollouts = 10000 # ロールアウト数
    z_dim = 32 # 潜在表現の次元数

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPUが使えるかどうかの確認

    model = VAE().to(device) # モデルをGPUへ
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # オプティマイザ

    train_history_r = [] # 学習用の損失関数(batchごとの平均)
    train_history_kl = [] # 学習用の損失関数
    train_history = [] # 学習用の損失関数
    test_history_r = [] # テスト用の損失関数(batchごとの平均)
    test_history_kl = [] # テスト用の損失関数
    test_history = [] # テスト用の損失関数

    env = CarRacing_rollouts() # rolloutの取得用

    for epoch in range(num_epoch):
        idx = np.arange(num_rollouts).astype(np.int32) # ロールアウトのインデックス
        train_idx = idx[:int(num_rollouts * 0.7)] # 学習用のロールアウトのインデックス
        test_idx = idx[int(num_rollouts * 0.7):] # テスト用のロールアウトのインデックス
        np.random.shuffle(train_idx) # 学習用のロールアウトのインデックスをシャッフル

        train_loss = 0 # 学習用の損失関数
        train_loss_r = 0 # 学習用の損失関数
        train_loss_kl = 0 # 学習用の損失関数
        test_loss = 0 # テスト用の損失関数
        test_loss_r = 0 # テスト用の損失関数
        test_loss_kl = 0 # テスト用の損失関数

        del idx

        model.train() # 学習モード
        print('Train Epoch: {}'.format(epoch+1))
        for i in range(0, len(train_idx), batch_size):
            if i + batch_size > len(train_idx):
                batch_idx = train_idx[i:]
            else:
                batch_idx = train_idx[i:i+batch_size]
            
            train_data, _, _, _ = env.load_rollouts(batch_idx) # ロールアウトの取得

            train_data = train_data / 255.0 # 値を0~1に正規化, [bar_size, seq_len, 64, 64, 3]
            train_data = train_data.reshape(-1, 3, 64, 64) # [batch_size, seq_len, 3, 64, 64] -> [batch_size * seq_len, 3, 64, 64]
            train_data = torch.tensor(train_data, dtype=torch.float32).to(device) # テンソル化, GPUへ

            optimizer.zero_grad() # 勾配の初期化
            y, mu, log_var = model(train_data) # z: [batch_size, z_dim], mu: [batch_size, z_dim], log_var: [batch_size, z_dim]
            loss, r_loss, kl_loss = model.loss(train_data, y, mu, log_var) # 損失関数の計算, スカラー値
            loss.backward() # 勾配の計算
            optimizer.step() # パラメータの更新

            train_loss += loss.item() # 損失関数の計算
            train_loss_r += r_loss.item() # 損失関数の計算
            train_loss_kl += kl_loss.item() # 損失関数の計算

            if i % 100 == 0:
                print('Epoch: {}, Batch: {}, Loss: {:.3f}, r_loss: {:.3f}, kl_loss: {:.3f}'.format(epoch+1, i, loss.item(), r_loss.item(), kl_loss.item()))

            del train_data
        
        train_history.append(train_loss / len(train_idx))
        train_history_r.append(train_loss_r / len(train_idx))
        train_history_kl.append(train_loss_kl / len(train_idx))

        del train_idx, train_loss, loss, y, mu, log_var, r_loss, kl_loss
        
        model.eval() # 評価モード
        print('Test Epoch: {}'.format(epoch+1))
        for i in range(0, len(test_idx), batch_size):
            if i + batch_size > len(test_idx):
                batch_idx = test_idx[i:]
            else:
                batch_idx = test_idx[i:i+batch_size]
            
            test_data, _, _, _ = env.load_rollouts(batch_idx)
            test_data = test_data / 255.0
            test_data = test_data.reshape(-1, 3, 64, 64)
            test_data = torch.tensor(test_data, dtype=torch.float32).to(device)

            y, mu, log_var = model(test_data)
            loss, r_loss, kl_loss = model.loss(test_data, y, mu, log_var)
            
            test_loss += loss.item()
            test_loss_r += r_loss.item()
            test_loss_kl += kl_loss.item()

            if i % 100 == 0:
                print('Epoch: {}, Batch: {}, Loss: {:.3f}, r_loss: {:.3f}, kl_loss: {:.3f}'.format(epoch+1, i, loss.item(), r_loss.item(), kl_loss.item()))
            
            del test_data

        test_history.append(test_loss / len(test_idx))
        test_history_r.append(test_loss_r / len(test_idx))
        test_history_kl.append(test_loss_kl / len(test_idx))

        del test_idx, test_loss, loss, y, mu, log_var

        # モデルの保存
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, './vae_tolerance.pth')
        
        # 学習用の損失関数の保存
        np.save('./train_history_torelance.npy', np.array(train_history))
        np.save('./train_history_r_torelance.npy', np.array(train_history_r))
        np.save('./train_history_kl_torelance.npy', np.array(train_history_kl))
        # テスト用の損失関数の保存
        np.save('./test_history_torelance.npy', np.array(test_history))
        np.save('./test_history_r_torelance.npy', np.array(test_history_r))
        np.save('./test_history_kl_torelance.npy', np.array(test_history_kl))

if __name__=="__main__":
    main()