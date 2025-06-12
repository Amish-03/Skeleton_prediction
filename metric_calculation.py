import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Settings ---
T, M, KEYPOINTS, DIM = 10, 5, 33, 3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GEN_PATH = 'generator_lstm1.pth'
REFERENCE_IDX = 0  # Reference joint (e.g. pelvis)
THRESH = 0.1       # Distance threshold for classification

# --- Generator Definition ---
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

# --- Load Model ---
G = Generator().to(DEVICE)
G.load_state_dict(torch.load(GEN_PATH, map_location=DEVICE))
G.eval()

# --- Load Data ---
X = np.load("X_pose_train.npy")
Y = np.load("Y_pose_train.npy")
X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
Y = torch.tensor(Y, dtype=torch.float32).to(DEVICE)

# --- Skeleton connections ---
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (12, 14), (14, 16), (11, 13), (13, 15), (15, 17),
    (12, 24), (11, 23), (24, 26), (26, 28), (23, 25), (25, 27)
]

def plot_skeleton(ax, pose, title='', color='blue'):
    pose = pose[:, :2]
    ax.scatter(pose[:, 0], -pose[:, 1], c=color, s=10)
    for a, b in BONES:
        if a < pose.shape[0] and b < pose.shape[0]:
            ax.plot([pose[a, 0], pose[b, 0]], [-pose[a, 1], -pose[b, 1]], color=color, linewidth=1)
    ax.set_title(title)
    ax.axis('equal')
    ax.axis('off')

# --- Evaluation Functions ---
def align_relative(pose, ref_idx=0):
    ref = pose[:, ref_idx, :]  # shape: (M, 3)
    return pose - ref[:, np.newaxis, :]

def compute_binary_map(real, pred, threshold=0.1):
    # Align both using relative distances
    real_rel = align_relative(real, REFERENCE_IDX)
    pred_rel = align_relative(pred, REFERENCE_IDX)

    # Euclidean distances between each point
    dist = np.linalg.norm(real_rel - pred_rel, axis=-1)  # (M, 33)
    binary_pred = (dist < threshold).astype(int)
    binary_true = np.ones_like(binary_pred)
    return binary_true.flatten(), binary_pred.flatten()

# --- Sample and Generate ---
idx = np.random.randint(0, len(X))
input_seq = X[idx:idx+1]
real_future = Y[idx:idx+1]
with torch.no_grad():
    gen_future = G(input_seq)

real_np = real_future[0].cpu().numpy()
pred_np = gen_future[0].cpu().numpy()

# --- Metrics Calculation ---
y_true, y_pred = compute_binary_map(real_np, pred_np, threshold=THRESH)
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

# --- Relative Errors ---
def relative_distances(skeleton_seq, ref_idx=0):
    ref = skeleton_seq[:, ref_idx, :]
    dists = np.linalg.norm(skeleton_seq - ref[:, np.newaxis, :], axis=-1)
    return dists

real_dists = relative_distances(real_np)
pred_dists = relative_distances(pred_np)
mae = np.abs(pred_dists - real_dists).mean()
mse = ((pred_dists - real_dists) ** 2).mean()
rmse = np.sqrt(mse)

# --- Print Metrics ---
print(f"\n📊 Relative Pose Evaluation (Threshold = {THRESH}):")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"MAE:       {mae:.6f}")
print(f"MSE:       {mse:.6f}")
print(f"RMSE:      {rmse:.6f}")

# --- Plotting ---
fig, axs = plt.subplots(2, M, figsize=(15, 6))
for i in range(M):
    plot_skeleton(axs[0, i], real_np[i], title=f"Real t+{i+1}", color='green')
    plot_skeleton(axs[1, i], pred_np[i], title=f"Fake t+{i+1}", color='red')

fig.suptitle("Top: Ground Truth | Bottom: Generated\n(Relative, Reference Keypoint = 0)", fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
