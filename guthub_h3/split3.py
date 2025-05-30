from Bio import SeqIO
import os

input_fasta = "H3_aligned.fasta"
output_dir = "split3"
group_size = 500

os.makedirs(output_dir, exist_ok=True)

records = list(SeqIO.parse(input_fasta, "fasta"))
total = len(records)
print(f"Total sequences: {total}")

for i in range(0, total, group_size):
    chunk = records[i:i + group_size]
    out_path = os.path.join(output_dir, f"split_{i // group_size + 1}.fasta")
    SeqIO.write(chunk, out_path, "fasta")
    print(f"Saved {len(chunk)} sequences to {out_path}")
