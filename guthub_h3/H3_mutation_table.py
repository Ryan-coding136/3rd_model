from Bio import SeqIO
import pandas as pd

fasta_path = "H3_aligned.fasta"
input_csv = "H3_id.csv"
output_csv = "H3_mutation_table_expanded.csv"

# site setting and aligment
mutation_sites = {
    137: 136,
    138: 137,
    190: 189,
    193: 192,
    225: 224,
    226: 225,
    228: 227
}

# referred AA
ref_aas = {
    137: "K",
    138: "S",
    190: "E",
    193: "Q",
    225: "G",
    226: "Q",
    228: "G"
}

# read isolated_id
df = pd.read_csv(input_csv)

# read fasta
seq_dict = {}
for record in SeqIO.parse(fasta_path, "fasta"):
    try:
        isolated_id = str(record.id).split("|")[1]
    except IndexError:
        isolated_id = str(record.id)
    seq_dict[isolated_id] = str(record.seq)

# add mutation
for pos, idx in mutation_sites.items():
    mut_col = f"Mut{pos}"
    orig_col = f"Original_AA_{pos}"
    df[mut_col] = 0
    df[orig_col] = "X"

    for i, row in df.iterrows():
        iso_id = row["Isolated_Id"]
        if iso_id in seq_dict and idx < len(seq_dict[iso_id]):
            aa = seq_dict[iso_id][idx]
            df.at[i, orig_col] = aa
            if aa != ref_aas[pos]:
                df.at[i, mut_col] = 1

# result
df.to_csv(output_csv, index=False)
print(f"mutation added and saved：{output_csv}")
