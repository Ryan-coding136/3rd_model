import pandas as pd
import numpy as np
import os

# Path
csv_path = "H3_balanced_training_set.csv"
embedding_dir = "split3"
id_dir = "split3/id"

output_X = "X.npy"
output_y = "y.npy"
output_voxel = "voxel.npy"

df = pd.read_csv(csv_path)
print(f"✅ Sample Num: {len(df)}")

#  embedding  {Isolated_Id: embedding}
embedding_map = {}

for i in range(1, 11):
    emb_file = os.path.join(embedding_dir, f"split_{i}_embeddings.npy")
    id_file = os.path.join(id_dir, f"split_{i}_ids.txt")
    if not os.path.exists(emb_file) or not os.path.exists(id_file):
        print(f"⚠️ Loss：{emb_file} or {id_file}，Skip")
        continue

    emb_array = np.load(emb_file)  # shape: (N, 1280)
    with open(id_file) as f:
        ids = [line.strip() for line in f]
    
    if len(ids) != emb_array.shape[0]:
        print(f"⚠️ ID num not matched：{id_file}")
        continue

    for eid, emb in zip(ids, emb_array):
        embedding_map[eid] = emb

print(f"✅ Embedding table：{len(embedding_map)} ")

# Final data
features = []
labels = []
voxels = []

for _, row in df.iterrows():
    eid = row["Isolated_Id"]
    if eid not in embedding_map:
        print(f"⚠️ skip：{eid} no embeddeing")
        continue

    emb = embedding_map[eid]
    struct_feat = row[[col for col in df.columns if col.startswith("RSA_") or col.startswith("Dist_") or col == "host_label"]].values.astype(np.float32)
    combined = np.concatenate([emb, struct_feat])  # (1295,)
    features.append(combined)
    labels.append(row["binding_score"])

    voxels.append(emb.reshape(1, 10,8,16))  # voxel reshape

X = np.array(features)
y = np.array(labels)
voxel = np.array(voxels)

np.save(output_X, X)
np.save(output_y, y)
np.save(output_voxel, voxel)

print(f"✅ Saved: X={X.shape}, y={y.shape}, voxel={voxel.shape}")
