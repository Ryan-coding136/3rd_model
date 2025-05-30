# download_h3pdb.py
import os
import urllib.request

pdb_ids = ["4O5N", "7KOA", "6AOV"]
save_dir = "pdb_h3_templates"
os.makedirs(save_dir, exist_ok=True)

for pdb_id in pdb_ids:
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    path = os.path.join(save_dir, f"{pdb_id}.pdb")
    urllib.request.urlretrieve(url, path)
    print(f"✅ : {pdb_id}")
