import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm
import random

# ===== Data =====
X = np.load("X.npy")
y = np.load("y.npy")
voxel = np.load("voxel.npy")

# ===== Seed =====
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===== Model Structure =====
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

# ===== Evaluation =====
seeds = [42, 2023, 7, 1024, 77]
results = []

for seed in tqdm(seeds, desc="Evaluating multiple seeds"):
    set_seed(seed)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    voxel_tensor = torch.tensor(voxel, dtype=torch.float32)

    X_train, X_val, voxel_train, voxel_val, y_train, y_val = train_test_split(
        X_tensor, voxel_tensor, y_tensor, test_size=0.2, stratify=y_tensor, random_state=seed)

    train_ds = TensorDataset(X_train, voxel_train, y_train)
    val_ds = TensorDataset(X_val, voxel_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = Hybrid3DCNN().cuda()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.2, 1.3]).cuda(), label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0
    patience = 8

    for epoch in range(50):
        model.train()
        for x_feat, x_voxel, y_true in train_loader:
            x_feat, x_voxel, y_true = x_feat.cuda(), x_voxel.cuda(), y_true.cuda()
            pred = model(x_voxel, x_feat)
            loss = criterion(pred, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 8
            best_logits = all_preds
            best_y_true = all_true
        else:
            patience -= 1
            if patience == 0:
                break

    # Final caculation
    y_prob = torch.softmax(best_logits, dim=1).numpy()
    y_true = best_y_true.numpy()
    auc = roc_auc_score(F.one_hot(torch.tensor(y_true), num_classes=3).numpy(), y_prob, average="macro", multi_class="ovr")
    report = classification_report(y_true, best_logits.argmax(dim=1).numpy(), output_dict=True, zero_division=0)
    results.append({"seed": seed, "val_acc": best_val_acc, "auc": auc, "f1_macro": report["macro avg"]["f1-score"]})

# Output
print("\n==== Multiseed Evaluation Result====")
for r in results:
    print(f"Seed: {r['seed']} | Acc: {r['val_acc']:.4f} | AUC: {r['auc']:.4f} | F1: {r['f1_macro']:.4f}")

accs = [r['val_acc'] for r in results]
aucs = [r['auc'] for r in results]
f1s  = [r['f1_macro'] for r in results]

print(f"\nAve Acu: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Ave AUC:   {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"Ave F1:    {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
