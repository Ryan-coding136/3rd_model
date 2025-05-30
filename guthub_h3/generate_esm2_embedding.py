import torch
import esm
import numpy as np
from Bio import SeqIO
from tqdm import tqdm

# path setteing
fasta_path = "/split3/split_1.fasta"
output_path = "/split3/split_1_embeddings.npy" #choose different fasta manually

# load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval().cuda()


sequences = [(record.id, str(record.seq)) for record in SeqIO.parse(fasta_path, "fasta")]


batch_size = 8  # adjustable
all_embeddings = []

# generate embeddings
for i in tqdm(range(0, len(sequences), batch_size), desc="Generating embeddings"):
    batch = sequences[i:i + batch_size]
    labels, strs, tokens = batch_converter(batch)
    tokens = tokens.cuda()

    with torch.no_grad():
        results = model(tokens, repr_layers=[33], return_contacts=False)
    representations = results["representations"][33]

    for j, (_, seq) in enumerate(batch):
        seq_len = len(seq)
        embedding = representations[j, 1:seq_len + 1].cpu().numpy()

        avg_embedding = embedding.mean(axis=0)
        all_embeddings.append(avg_embedding)

np.save(output_path, np.array(all_embeddings))
print(f"✅ Saved to: {output_path} | Shape: {np.array(all_embeddings).shape}")
