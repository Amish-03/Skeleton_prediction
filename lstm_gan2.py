import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os

# --- Hyperparameters ---
T, M, KEYPOINTS, DIM = 10, 5, 33, 3
EPOCHS = 1000
BATCH_SIZE = 8
LR = 0.0002
LAMBDA_L1 = 10.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GEN_PATH = 'generator_lstm1.pth'
DISC_PATH = 'discriminator_lstm1.pth'

# --- Load Data ---
X = np.load('X_pose_train.npy')
Y = np.load('Y_pose_train.npy')
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)
loader = DataLoader(TensorDataset(X, Y), batch_size=BATCH_SIZE, shuffle=True)

# --- Generator ---
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(KEYPOINTS * DIM, 512),
            nn.LayerNorm(512),
            nn.SiLU()
        )
        self.lstm = nn.LSTM(512, 512, 2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, M * KEYPOINTS * DIM)
        )

    def forward(self, x):
        b = x.size(0)
        x = self.input_proj(x.view(b, T, -1))
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1]).view(b, M, KEYPOINTS, DIM)

# --- Discriminator ---
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(KEYPOINTS * DIM, 512),
            nn.LayerNorm(512),
            nn.SiLU()
        )
        self.lstm = nn.LSTM(512, 512, 2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)  # No sigmoid
        )

    def forward(self, y):
        b = y.size(0)
        y = self.input_proj(y.view(b, M, -1))
        _, (hn, _) = self.lstm(y)
        return self.fc(hn[-1])

# --- Gradient Penalty ---
def gradient_penalty(D, real, fake):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=DEVICE)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_inter = D(interpolated)
    ones = torch.ones_like(d_inter)
    grad = torch.autograd.grad(
        outputs=d_inter,
        inputs=interpolated,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grad = grad.view(grad.size(0), -1)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()

# --- Initialize ---
G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)
if os.path.exists(GEN_PATH): G.load_state_dict(torch.load(GEN_PATH)); print("🔁 Loaded Generator checkpoint")
if os.path.exists(DISC_PATH): D.load_state_dict(torch.load(DISC_PATH)); print("🔁 Loaded Discriminator checkpoint")

# --- Loss and Optimizers ---
loss_bce = nn.BCEWithLogitsLoss()
loss_l1 = nn.L1Loss()
opt_g = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_d = optim.Adam(D.parameters(), lr=LR / 5, betas=(0.5, 0.999))

# --- Training ---
for epoch in range(EPOCHS):
    G.train(); D.train()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        b = xb.size(0)
        real = torch.ones((b, 1), device=DEVICE)
        fake = torch.zeros((b, 1), device=DEVICE)

        # --- Discriminator ---
        with torch.no_grad():
            gen_y = G(xb)
        d_real = D(yb)
        d_fake = D(gen_y.detach())
        gp = gradient_penalty(D, yb, gen_y.detach())
        loss_d = loss_bce(d_real, real) + loss_bce(d_fake, fake) + 10 * gp
        D.zero_grad(); loss_d.backward(); opt_d.step()

        # --- Generator ---
        gen_y = G(xb)
        pred_fake = D(gen_y)
        loss_g_adv = loss_bce(pred_fake, real)
        loss_g_l1 = loss_l1(gen_y, yb)
        loss_g = (loss_g_adv + LAMBDA_L1 * loss_g_l1) if epoch >= 10 else LAMBDA_L1 * loss_g_l1
        G.zero_grad(); loss_g.backward(); opt_g.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | D: {loss_d.item():.4f} | G: {loss_g.item():.4f} | L1: {loss_g_l1.item():.4f}")

    if (epoch + 1) % 10 == 0:
        torch.save(G.state_dict(), GEN_PATH)
        torch.save(D.state_dict(), DISC_PATH)
        print(f"💾 Checkpoint saved at epoch {epoch+1}")

print("✅ Training complete. Models saved.")
