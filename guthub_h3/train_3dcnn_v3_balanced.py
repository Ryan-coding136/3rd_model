import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

# ===== Seed =====
def set_seed(seed=40):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(40)

# ===== Setting =====
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
PATIENCE = 8
ALPHA = torch.tensor([1.0, 1.2, 1.3])
LABEL_SMOOTHING = 0.1

# ===== Data =====
X = np.load("X.npy")
y = np.load("y.npy")
voxel = np.load("voxel.npy")

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
voxel_tensor = torch.tensor(voxel, dtype=torch.float32)

X_train, X_val, voxel_train, voxel_val, y_train, y_val = train_test_split(
    X_tensor, voxel_tensor, y_tensor, test_size=0.2, stratify=y_tensor, random_state=42)

train_ds = TensorDataset(X_train, voxel_train, y_train)
val_ds = TensorDataset(X_val, voxel_val, y_val)

def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=worker_init_fn)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ===== Model =====
class Hybrid3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.mlp = nn.Sequential(
            nn.Linear(16 + 1295, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )

    def forward(self, x_voxel, x_feat):
        cnn_out = self.cnn(x_voxel).view(x_voxel.size(0), -1)
        combined = torch.cat([cnn_out, x_feat], dim=1)
        return self.mlp(combined)

model = Hybrid3DCNN().cuda()
criterion = nn.CrossEntropyLoss(weight=ALPHA.cuda(), label_smoothing=LABEL_SMOOTHING)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ===== Training Process =====
best_val_acc = 0
patience = PATIENCE

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for x_feat, x_voxel, y_true in train_loader:
        x_feat, x_voxel, y_true = x_feat.cuda(), x_voxel.cuda(), y_true.cuda()
        pred = model(x_voxel, x_feat)
        loss = criterion(pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for x_feat, x_voxel, y_true in val_loader:
            x_feat, x_voxel = x_feat.cuda(), x_voxel.cuda()
            logits = model(x_voxel, x_feat)
            all_preds.append(logits.cpu())
            all_true.append(y_true)

    all_preds = torch.cat(all_preds)
    all_true = torch.cat(all_true)
    pred_labels = all_preds.argmax(dim=1)
    val_acc = (pred_labels == all_true).float().mean().item()

    print(f"\U0001F4D8 Epoch {epoch} | Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience = PATIENCE
        torch.save(model.state_dict(), "best_model.pt")
        print("✅ Saved new best model")
    else:
        patience -= 1
        if patience == 0:
            print("⏹️ Early stopping triggered.")
            break

# ===== Evaluation =====
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
with torch.no_grad():
    val_logits = model(val_ds[:][1].cuda(), val_ds[:][0].cuda())
    y_pred = val_logits.argmax(dim=1).cpu()
    y_prob = torch.softmax(val_logits, dim=1).cpu()
    y_true = val_ds[:][2]

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2], yticklabels=[0,1,2])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_v3_stable.png")
print("🖼️ Confusion matrix saved to confusion_matrix_v3_stable.png")

# ROC Curve
y_true_bin = F.one_hot(y_true, num_classes=3).numpy()
fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob.numpy().ravel())
auc_score = roc_auc_score(y_true_bin, y_prob.numpy(), average="macro", multi_class="ovr")
plt.figure()
plt.plot(fpr, tpr, label=f"Macro AUC = {auc_score:.3f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_v3_stable.png")
print("📈 ROC curve saved to roc_curve_v3_stable.png")
