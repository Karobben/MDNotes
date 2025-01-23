---
toc: true
url: esm
covercopy: © Karobben
priority: 10000
date: 2025-01-22 10:06:18
title: "esm, Evolutionary Scale Modeling"
ytitle: "esm, Evolutionary Scale Modeling"
description: "esm, Evolutionary Scale Modeling"
excerpt: "ESM (Evolutionary Scale Modeling) is a family of large-scale protein language models developed by Meta AI. They’re trained on massive protein sequence databases, learning contextual representations of amino acids purely from sequence data. These representations—often called embeddings—capture both structural and functional clues.<br>In practice, you feed a protein sequence into ESM to obtain per-residue embeddings, which you can then use for downstream tasks like structure prediction, function annotation, or variant effect prediction. If you batch multiple sequences together, ESM aligns them by adding special start/end tokens and padding shorter sequences to match the longest one. You then slice out the valid embedding region for each protein, ignoring any padding."
tags: [Bioinformatics, Biochmistry, Biology, AI]
category: [Biology, Bioinformatics, Protein Structure]
cover: "https://user-images.githubusercontent.com/3605224/199301187-a9e38b3f-71a7-44be-94f4-db0d66143c53.png"
thumbnail: "https://user-images.githubusercontent.com/3605224/199301187-a9e38b3f-71a7-44be-94f4-db0d66143c53.png"
---

## Basic Use

```python
import torch
import esm

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
data = [
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein3",  "K A <mask> I S Q"),
]
batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

# Extract per-residue representations (on CPU)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
token_representations = results["representations"][33]

# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
sequence_representations = []
for i, tokens_len in enumerate(batch_lens):
    sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

# Look at the unsupervised self-attention map contact predictions
import matplotlib.pyplot as plt
for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
    plt.matshow(attention_contacts[: tokens_len, : tokens_len])
    plt.title(seq)
    plt.show()
```

## Get the Embedding for Each Residues

**ESM** (like many transformer-based models) uses “special tokens” plus padding so that all sequences in a batch have the same length. Specifically:

1. **Start and End Tokens**: For any single sequence of length \( n \), the ESM model prepends a start token and appends an end token. That gives you \( n + 2 \) positions for a single sequence.

2. **Batch Processing Requires Padding**: When you process multiple sequences in a single batch, they all get padded (on the right) to match the length of the _longest_ sequence in the batch. So if the longest sequence has \( n \) residues, _all_ sequences become length \( n + 2 \) (including the special tokens), and shorter sequences get padding tokens to fill in the gap.

Hence, whether a sequence originally has \( k \) residues or \( m \) residues, in a batch whose _longest_ sequence is \( n \) residues, everyone ends up with a vector length of \( n + 2 \). This ensures the entire input tensor in the batch has a uniform shape.

Here is an example of extract the embedding by following codes above:

```python
Seq_Embeding = {i[0]:token_representations[0][:len(i[1])+2] for i,ii in zip(data,token_representations) }
# also, this is for remove the start and end
Seq_Embeding = {i[0]:token_representations[0][1:len(i[1])+1] for i,ii in zip(data,token_representations) }
```

The embedding results form batch and single chain are general the same but slightly different. If you embedding them one by one and calculate the difference, you'll find there are slight different. According to the ChatGPT, it could be caused by:

1. **Position Embeddings**  
   - ESM (like most Transformer models) uses positional embeddings. If the model sees a “longer” padded batch, the position indices for each token can differ from the single-sequence scenario, so the sequence’s tokens may be mapped to slightly different (learned) position embeddings.  

2. **Attention Masking and Context**  
   - In a batched setting, the model creates a larger attention mask (covering all tokens up to the longest sequence in the batch). Although it’s not supposed to mix information across sequences, the internal computations (e.g., how attention is batched or chunked) can differ from the single-sequence forward pass, leading to small numeric discrepancies.

3. **Dropout or Other Stochastic Layers**  
   - If your model isn’t in `eval()` mode (or if dropout is enabled for any reason), you’ll get random differences each pass. Always ensure `model.eval()` and (ideally) a fixed random seed for more reproducible outputs.

4. **Floating-Point Rounding**  
   - GPU parallelization can cause minor floating-point differences, especially between batched and single-inference calls. These are typically very small numerical deviations.





<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>
