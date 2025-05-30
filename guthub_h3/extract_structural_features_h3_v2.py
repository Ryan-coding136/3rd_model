import os
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser, DSSP

# path
pdb_dir = "pdb_h3_templates"
input_csv = "H3_mutation_table_expanded.csv"
output_csv = "H3_mutation_table_structural_v2.csv"

# 7 key mutations
key_positions = [137, 138, 190, 193, 225, 226, 228]

df = pd.read_csv(input_csv)

rsa_all, dist_all = [], []
parser = PDBParser(QUIET=True)
pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith(".pdb")]

for pdb_file in pdb_files:
    pdb_id = pdb_file.replace(".pdb", "")
    structure = parser.get_structure(pdb_id, os.path.join(pdb_dir, pdb_file))
    model = structure[0]
    chain = list(model.get_chains())[0]

    try:
        dssp = DSSP(model, os.path.join(pdb_dir, pdb_file))
    except Exception as e:
        print(f"[DSSP Error] {pdb_id}: {e}")
        continue

    ca_coords = {}
    for res in chain:
        res_id = res.get_id()[1]
        if res_id in key_positions:
            if "CA" in res:
                ca_coords[res_id] = res["CA"].get_coord()
            try:
                rsa = dssp[(chain.id, res_id)][3]
                rsa_all.append({"PDB": pdb_id, "Position": res_id, "RSA": rsa})
            except Exception:
                pass

    for i in key_positions:
        for j in key_positions:
            if i != j and i in ca_coords and j in ca_coords:
                dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
                dist_all.append({
                    "PDB": pdb_id,
                    "Pos1": i,
                    "Pos2": j,
                    "Distance": dist
                })

rsa_df = pd.DataFrame(rsa_all)
dist_df = pd.DataFrame(dist_all)
avg_rsa = rsa_df.groupby("Position")["RSA"].mean().reset_index()
avg_rsa.columns = ["Position", "Avg_RSA"]
avg_dist = dist_df.groupby(["Pos1", "Pos2"])["Distance"].mean().reset_index()
avg_dist.columns = ["Pos1", "Pos2", "Avg_Dist"]

for pos in key_positions:
    rsa_val = avg_rsa[avg_rsa["Position"] == pos]["Avg_RSA"].values[0]
    dist_rows = avg_dist[(avg_dist["Pos1"] == pos) | (avg_dist["Pos2"] == pos)]
    dist_val = dist_rows["Avg_Dist"].mean()

    df[f"RSA_{pos}"] = rsa_val
    df[f"Dist_{pos}"] = dist_val

df.to_csv(output_csv, index=False)
print(f"✅ Saved：{output_csv}")
