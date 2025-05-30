import pandas as pd

def extract_ids_from_fasta(fasta_path):
    ids = []
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                parts = line.strip().split('|')
                if len(parts) > 1:
                    ids.append(parts[1])
    return ids


metadata_path = "/mnt/e/H3/matched_metadata.csv"
fasta_path = "/mnt/e/H3//H3_aligned.fasta"
output_path = "/mnt/e/H3/H3_metadata.csv"
missing_output = "/mnt/e/H3/H3_missing_ids.txt"

# Step 1
fasta_ids = extract_ids_from_fasta(fasta_path)
print(f" Extract {len(fasta_ids)} ID")

# Step 2
metadata_df = pd.read_csv(metadata_path)
isolated_col = metadata_df.columns[0]
print(f" Use '{isolated_col}' to match")

# Step 3
filtered_df = metadata_df[metadata_df[isolated_col].isin(fasta_ids)]
print(f"✅ Matched：{len(filtered_df)} ")

# Step 4
fasta_id_set = set(fasta_ids)
metadata_id_set = set(metadata_df[isolated_col].astype(str))

missing_ids = sorted(fasta_id_set - metadata_id_set)
print(f"❗ Missing ID：{len(missing_ids)}")

# Step 5:
filtered_df.to_csv(output_path, index=False)
print(f"✅ Saved：{output_path}")

# Step 6:
with open(missing_output, 'w') as f:
    for mid in missing_ids:
        f.write(mid + '\n')
print(f"📄 Missing ID saved：{missing_output}")
