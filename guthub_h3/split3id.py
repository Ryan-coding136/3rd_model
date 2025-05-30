from Bio import SeqIO
import os

fasta_dir = "split3"
output_ids_dir = "split3/id"

for i in range(1, 11):
    fasta_file = os.path.join(fasta_dir, f"split_{i}.fasta")
    output_file = os.path.join(output_ids_dir, f"split_{i}_ids.txt")

    ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        full_id = record.id
        parts = full_id.split("|")
        epi_id = None
        for part in parts:
            if part.startswith("EPI_ISL_"):
                epi_id = part
                break
        if epi_id:
            ids.append(epi_id)
        else:
            print(f"⚠️ Unable to extract EPI ID from header : {full_id}")

    with open(output_file, "w") as f:
        for eid in ids:
            f.write(eid + "\n")

    print(f"✅  {len(ids)}  ID → {output_file}")
