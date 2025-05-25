#  H3 HA Binding Affinity Prediction Pipeline

This repository contains a complete pipeline for predicting the receptor-binding affinity of **H3 subtype Hemagglutinin (HA) proteins** using a hybrid deep learning model that integrates sequence embeddings, structural features, and host metadata.

---

##  Pipeline Overview

**Goal**: Classify each HA protein sequence into one of three binding affinity classes:
- `0`: Non-binding
- `1`: Weak binding
- `2`: Strong binding

**Core idea**: Combine ESM-2 protein language model embeddings, voxelized 3D representations, and structure-informed features from PDB.

---

##  Pipeline Components

### 1. **Data Preparation**
- Input: `H3_aligned.fasta`
- Split into batches of 500 sequences per file

### 2. **ESM-2 Embedding Generation**
- Generate per-sequence embeddings using `esm2_t33_650M_UR50D`
- Embedding shape: `(L, 1280)` where L = sequence length
- Reshape global embeddings into voxel format `(1, 10, 8, 16)` for 3D CNN

### 3. **Structural Feature Extraction**
- Source: PDB structures `4O5N`, `7KOA`, `6AOV`
- Extracted features:
  - RSA (Relative Solvent Accessibility)
  - Euclidean distance between mutation sites
- Aggregated into average structural matrices for each residue position

### 4. **Binding Score Labeling**
- Rule-based scoring using:
  - Mutation presence at key positions: 193, 226, 228, etc.
  - RSA > 0.25 → likely surface-exposed
  - Distance < 12 Å → near receptor site
- Final label assignment based on total structural score:
  - `0`: score ≤ 5.0
  - `1`: 5.0 < score ≤ 10.0
  - `2`: score > 10.0

### 5. **Host Metadata Integration**
- Source: `H3_metadata.csv`
- Label: `host_label = 0` (human), `1` (non-human)
- Merged into final dataset `H3_mutation_table_labeled_with_host.csv`

### 6. **Dataset Construction**
- Inputs:
  - Sequence embedding (1295 features)
  - Voxel: `(1, 10, 8, 16)`
- Balanced training set: ~400 samples per class
- Final files:
  - `X.npy` → features
  - `y.npy` → labels
  - `voxel.npy` → voxel inputs

---

##  Model Architecture

**Hybrid 3D CNN**:
- 3D Convolution over voxel input
- MLP over concatenated [CNN features + sequence features]
- Loss: `CrossEntropyLoss(weight=[1.0, 1.2, 1.3], label_smoothing=0.1)`

---

## Evaluation

### Multi-seed robustness
- Seeds: 42, 2023, 7, 1024, 77

| Metric       | Mean ± Std          |
|--------------|---------------------|
| Accuracy     | 0.7491 ± 0.0941     |
| Macro AUC    | 0.8638 ± 0.0614     |
| Macro F1     | 0.7481 ± 0.1169     |

### Confusion matrix and ROC curve saved as:
- `confusion_matrix_v3.png`
- `roc_curve_v3.png`

---

##  Next Steps

- Improve class 1 (weak binding) recall
- Introduce focal loss or additional class balancing
- Extend to other subtypes (e.g. H5, H7)
- Grad-CAM/SHAP interpretation of mutation impact

---

##  Folder Structure

```bash
.
├── split3/                   # ESM-2 embeddings
├── structure_features/       # RSA & Dist matrices
├── H3_mutation_table_labeled_with_host.csv
├── X.npy / y.npy / voxel.npy # Final training dataset
├── train_3dcnn.py            # Main training script
├── multi_seed_eval.py        # Multi-seed evaluation
```

---

##  Citation & Credits
- ESM-2: [https://github.com/facebookresearch/esm](https://github.com/facebookresearch/esm)
- PDB structures via [https://rcsb.org](https://rcsb.org)

---

