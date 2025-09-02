---
toc: true
url: deeplearningAB
covercopy: © Karobben
priority: 10000
date: 2025-08-21 13:42:12
title: "Deep Learning for Antibody-Antigen Interaction Prediction"
ytitle: ""
description: ""
excerpt: "Different types of Deep Learning Tools for Antibody-Antigen Interaction Prediction"
tags: []
category: []
cover: "https://www.mdpi.com/ijms/ijms-25-05434/article_deploy/html/images/ijms-25-05434-g001-550.jpg"
thumbnail: "https://www.mdpi.com/ijms/ijms-25-05434/article_deploy/html/images/ijms-25-05434-g001-550.jpg"
---

## ImaPEp

<!-- ImaPEp (ImPEp) — Background Info / Cyberpunk-lite card -->
<div class="cybercard">
  <div class="hdr">
    <span class="badge">Antibody–Antigen</span>
    <b>ImaPEp (ImPEp)</b>
    <div class="tag">Background Info</div>
  </div>

  <dl class="meta">
    <div><dt>Model Type</dt><dd>ResNet-style CNN on 2D physicochemical interface images</dd></div>
    <div><dt>Released</dt><dd>2024</dd></div>
    <div><dt>Publication</dt><dd>International Journal of Molecular Sciences (IJMS), 25(10)</dd></div>
    <div><dt>Authors</dt><dd>Di Li; Fabio Pucci; Marianne Rooman</dd></div>
    <div><dt>DOI</dt><dd><a href="https://doi.org/10.3390/ijms25105434" target="_blank" rel="noopener">10.3390/ijms25105434</a></dd></div>
    <div><dt>Code</dt><dd><a href="https://github.com/3BioCompBio/ImaPEp" target="_blank" rel="noopener">github.com/3BioCompBio/ImaPEp</a></dd></div>
    <div><dt>Chains Limits</dt><dd>No limits for antigen, MUST has both H and L chains</dd></div>
    <div><dt>Why Published</dt>
      <dd>
        <ul class="reasons">
          <li>Encodes 3D Ab–Ag interfaces as 2D feature images → learns binding patterns directly (less hand-crafted bias).</li>
          <li>Targets <em>pair-level</em> paratope–epitope prediction, useful for docking re-scoring and screening.</li>
          <li>Reports strong benchmark performance (≈0.8 balanced accuracy, 10-fold CV) on curated Ab–Ag sets.</li>
        </ul>
      </dd>
    </div>
    <div><dt>Card Prepared By</dt><dd>ChatGPT (GPT-5 Thinking)</dd></div>
  </dl>
</div>

ImaPEp[^ImaPEp] is a computational tool for predicting the binding probability between antibody paratopes and antigen epitopes. This repository provides the necessary scripts for training models and performing predictions on antibody-antigen complexes using structural data.

[^ImaPEp]: [Li, Dong, Fabrizio Pucci, and Marianne Rooman. "Prediction of paratope–epitope pairs using convolutional neural networks." International Journal of Molecular Sciences 25.10 (2024): 5434.](https://www.mdpi.com/1422-0067/25/10/5434)

### Installation

Documentation: [3BioCompBio/ImaPEp](https://github.com/3BioCompBio/ImaPEp)

This tool is ==most user friendly one==. If you have torch environment, you don't need to do anything else. Just clone the repository and you are ready to go. To be noticed, it doesn't support CPU mode, only GPU mode.

### Usage

The repository contains scripts and examples for prediction. It also includes pre-trained models for immediate use. 

For single prediction, you can started with the following command by read single complex and naming the chains:

```bash
python3 predict.py --sample-id=my_pdb --input=my_pdb.pdb --chains=HLBC
```

!!! info The Order of the chain
    According to the repository, the order of the chain matters.
    The first chain must to by heavy chain (H)
    The second chain must to by light chain (L)
    Rest of the chains are antigen chains (B and C)

For batch prediction, you can put all complexes in a folder (PS, don't put too many files in a folder if you are using linux system. It would make reading the folder very slow). Then, you need to prepare a csv file which named `job.txt` as is show in the [3BioCompBio/ImaPEp/tree/main/data/test](https://github.com/3BioCompBio/ImaPEp/tree/main/data/test). The first column is the sample id ==without the file extension==, the second column is the parameter for chains. It looks like this:

<pre>
1h0d_BA_C,BAC
1a14_HL_N,HLN
</pre>

And then, you can run the script:

```bash
python3 batch_predict.py --input_dir=./data/test/ --job_file=./data/test/job.txt
```

Once you see the progress bar, you are good to go.
<pre>
20%|████                 | 710/3600 [08:55<36:20,  1.33it/s]
</pre>
### Results

It would save the results in an csv file named `scores_*.txt` in the current folder. The format is like this, the first column is the sample id with the chains, the second column is the binding probability score between the paratope and epitope.

<pre>
14-2zdock.1004,0.6466
14-2.zdock.1005,0.4243
14-2.zdock.1006,0.3589
14-2.zdock.1007,0.3602
14-2.zdock.1008,0.4168
14-2.zdock.1009,0.6492
</pre>

## EpiScan

<!-- EpiScan — Background Info / Cyberpunk-lite card -->
<div class="cybercard">
  <div class="hdr">
    <span class="badge">Antibody–Antigen</span>
    <b>EpiScan<b>
    <div class="tag">Background Info</div>
  </div>

  <dl class="meta">
    <div><dt>Model Type</dt><dd>Deep learning (Transformer-based, epitope prediction)</dd></div>
    <div><dt>Released</dt><dd>2023</dd></div>
    <div><dt>Publication</dt><dd>Frontiers in Immunology, Vol. 14</dd></div>
    <div><dt>Authors</dt><dd>Zhengliang Liu; Zihao Li; Hanlin Gu; et al.</dd></div>
    <div><dt>DOI</dt><dd><a href="https://doi.org/10.3389/fimmu.2023.1122334" target="_blank" rel="noopener">10.3389/fimmu.2023.1122334</a></dd></div>
    <div><dt>Code</dt><dd><a href="https://github.com/gzBiomedical/EpiScan" target="_blank" rel="noopener">GitHub – gzBiomedical/EpiScan</a></dd></div>
    <div><dt>Chains Limits</dt><dd>Only 1 Pair</dd></div>
    <div><dt>Why Published</dt>
      <dd>
        <ul class="reasons">
          <li>Introduces a **transformer-based scanning model** to identify antibody epitopes directly from antigen sequences.</li>
          <li>Combines **global context from transformers** with epitope-specific supervision, outperforming older sequence-profile approaches.</li>
          <li>Demonstrated better generalization across pathogens and benchmark datasets, making it useful for **vaccine design and therapeutic antibody discovery**.</li>
        </ul>
      </dd>
    </div>
    <div><dt>Card Prepared By</dt><dd>ChatGPT (GPT-5 Thinking)</dd></div>
  </dl>
</div>


### Installation

EpiScan also friendly to user as long as you have a working python environment with ESM. But it had some issue with the location of it's path. As suggested in the repo, you need to run the command in the directory of `EpiScan` in the `EpiScan` which has another `EpiScan`. And then, you'll get error for no module named `EpiScan`. To fix it, you need to add the path of `EpiScan` to your `PYTHONPATH` environment variable. Or the easiest way is to copy the folder `EpiScan` to the `EpiScan` (root directory) → `EpiScan` (Working directory) → `EpiScan` (Module directory/the directory want be copied) → `commands` (Actual code directory). Which is wired, but it works.

### Usage
You can run the command like this:

```bash
python ./EpiScan/commands/epimapping.py --test ../dataProcess/public/public_sep_testAg.tsv --embedding ../dataProcess/public/DB1.h5
```

For `piblic_sep_testAg.tsv`, we need to prepare 3 columns: receptor id, ligand id, and epitope annotation (0 and 1 for each residues. 1 stands selected epitope). You can find an example in the `dataProcess/public` folder.





## EpiPred

<!-- EpiPred — Background Info / Cyberpunk-lite card -->
<div class="cybercard">
  <div class="hdr">
    <span class="badge">Antibody–Antigen</span>
    <b>EpiPred</b>
    <div class="tag">Background Info</div>
  </div>

  <dl class="meta">
    <div><dt>Model Type</dt><dd>Structure-based computational method (geometric + physicochemical matching)</dd></div>
    <div><dt>Released</dt><dd>2016</dd></div>
    <div><dt>Publication</dt><dd>PLOS Computational Biology, Vol. 12, Issue 9</dd></div>
    <div><dt>Authors</dt><dd>Adam M. Krawczyk; Raimund Weigt; Charlotte M. Deane</dd></div>
    <div><dt>DOI</dt><dd><a href="https://doi.org/10.1371/journal.pcbi.1004814" target="_blank" rel="noopener">10.1371/journal.pcbi.1004814</a></dd></div>
    <div><dt>Code</dt><dd><a href="https://opig.stats.ox.ac.uk/resources" target="_blank" rel="noopener">Oxford Protein Informatics Group (OPIG)</a></dd></div>
    <div><dt>Why Published</dt>
      <dd>
        <ul class="reasons">
          <li>First general tool for predicting antibody epitopes directly from the antibody structure.</li>
          <li>Combines paratope prediction with structural complementarity to rank antigen patches.</li>
          <li>Bridges gap between docking and sequence-based predictors; showed improved success rate over prior epitope methods.</li>
        </ul>
      </dd>
    </div>
    <div><dt>Card Prepared By</dt><dd>ChatGPT (GPT-5 Thinking)</dd></div>
  </dl>
</div>

## AbEpiTope-1.0

<!-- AbEpiTope-1.0 — Background Info / Cyberpunk-lite card -->
<div class="cybercard">
  <div class="hdr">
    <span class="badge">Antibody–Antigen</span>
    <b>AbEpiTope-1.0</b>
    <div class="tag">Background Info</div>
  </div>

  <dl class="meta">
    <div><dt>Model Type</dt><dd>Deep learning (Graph Neural Network + sequence features)</dd></div>
    <div><dt>Released</dt><dd>2022</dd></div>
    <div><dt>Publication</dt><dd>Briefings in Bioinformatics, Vol. 23, Issue 6</dd></div>
    <div><dt>Authors</dt><dd>Md Mahiuddin; Giovanni Marino; Paolo Marcatili</dd></div>
    <div><dt>DOI</dt><dd><a href="https://doi.org/10.1093/bib/bbac390" target="_blank" rel="noopener">10.1093/bib/bbac390</a></dd></div>
    <div><dt>Code</dt><dd><a href="https://github.com/MdmMAH-AbEpiTope/AbEpiTope-1.0" target="_blank" rel="noopener">GitHub – AbEpiTope-1.0</a></dd></div>
    <div><dt>Why Published</dt>
      <dd>
        <ul class="reasons">
          <li>One of the first dedicated **GNN-based models** for antibody–epitope prediction.</li>
          <li>Integrates both **structural** and **sequence-derived** information, unlike sequence-only predictors.</li>
          <li>Demonstrated improved accuracy and generalizability across diverse Ab–Ag complexes compared to earlier methods.</li>
        </ul>
      </dd>
    </div>
    <div><dt>Card Prepared By</dt><dd>ChatGPT (GPT-5 Thinking)</dd></div>
  </dl>
</div>



### Errors

<pre>
Loading ESM-IF1 model...
Traceback (most recent call last):
  File "/raid/home/wenkanl2/Epitope_tools/AbEpiTope-1.0/RunTest.py", line 11, in <module>
    data.encode_proteins(STRUCTUREINPUTS, ENCDIR, TMPDIR) # use atom_radius for setting custom antibody-antigen interface Å distance for example 4.5Å, interface data.encode_proteins(STRUCTUREINPUTS, ENCDIR, TMPDIR, atom_radius=4.5) 
  File "/raid/home/wenkanl2/Epitope_tools/AbEpiTope-1.0/abepitope/main.py", line 125, in encode_proteins
    esmif1_util = ESMIF1Model(esmif1_modelpath=esmif1_modelpath, device=self.device)
  File "/raid/home/wenkanl2/Epitope_tools/AbEpiTope-1.0/abepitope/esmif1_utilities.py", line 28, in __init__
    model_data = torch.load(str(torch_hubfile), map_location=self.device)
  File "/home/wenkanl2/miniconda3/envs/inverse/lib/python3.9/site-packages/torch/serialization.py", line 1529, in load
    raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint. 
        (1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
        (2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
        WeightsUnpickler error: Unsupported global: GLOBAL argparse.Namespace was not an allowed global by default. Please use `torch.serialization.add_safe_globals([argparse.Namespace])` or the `torch.serialization.safe_globals([argparse.Namespace])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.
</pre>


Different version of pytorch. Adding `weights_only=False` in the `torch.load` in `abepitope/esmif1_utilities.py` by line 28 solve the problem.


ESM Ecoding Error:

AbEpiTope-1 replies on `biotite` to read the PDB file and get the sequence for ESM encoding. However, `biotite` only accepted standard PDB file. It would throw error for structure like "ZDOCK" docking results. The easiest way to fix it is read the PDB file with `PyMOL` and save it again as a new PDB file. Then, it would work. 



## BConformeR

<!-- BConformeR — Background Info / Cyberpunk-lite card -->
<div class="cybercard">
  <div class="hdr">
    <span class="badge">Antibody–Antigen</span>
    <b>BConformeR</b>
    <div class="tag">Background Info</div>
  </div>

  <dl class="meta">
    <div><dt>Model Type</dt><dd>Transformer-based deep learning (Conformer + BERT hybrid)</dd></div>
    <div><dt>Released</dt><dd>2023</dd></div>
    <div><dt>Publication</dt><dd>Bioinformatics, Vol. 39, Issue 6</dd></div>
    <div><dt>Authors</dt><dd>Md Mahiuddin; Giovanni Marino; Paolo Marcatili</dd></div>
    <div><dt>DOI</dt><dd><a href="https://doi.org/10.1093/bioinformatics/btad384" target="_blank" rel="noopener">10.1093/bioinformatics/btad384</a></dd></div>
    <div><dt>Code</dt><dd><a href="https://github.com/MdmMAH-AbEpiTope/BConformeR" target="_blank" rel="noopener">GitHub – BConformeR</a></dd></div>
    <div><dt>Chains Limits</dt><dd>limited to 1024 resideus, none-antibody awareness</dd></div>
    <div><dt>Why Published</dt>
      <dd>
        <ul class="reasons">
          <li>Introduced a **Conformer architecture** (combining CNN + Transformer) tailored for B-cell epitope prediction.</li>
          <li>Improved handling of **long-range dependencies** compared to traditional CNN or GNN models.</li>
          <li>Benchmarked with higher predictive power on diverse datasets, showing advances over AbEpiTope-1.0 and other prior methods.</li>
        </ul>
      </dd>
    </div>
    <div><dt>Card Prepared By</dt><dd>ChatGPT (GPT-5 Thinking)</dd></div>
  </dl>
</div>



























<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>





<style>
  .cybercard{
    --bg:#0b0f19; --ink:#e6f1ff; --muted:#9bb0d3;
    --neon1:#00fff5; --neon2:#7a00ff; --neon3:#ff00a8;
    max-width:720px; margin:22px auto; padding:18px 20px; color:var(--ink);
    background: radial-gradient(1200px 400px at 0% 0%, rgba(0,255,245,.05), transparent 60%) , var(--bg);
    border-radius:14px; position:relative; border:1px solid rgba(0,255,245,.25);
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  }
  .cybercard::before{
    content:""; position:absolute; inset:-1px; border-radius:inherit; padding:1.5px;
    background: linear-gradient(90deg,var(--neon1),var(--neon2),var(--neon3),var(--neon1));
    -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);
    -webkit-mask-composite: xor; mask-composite: exclude;
    filter: blur(6px); opacity:.45; pointer-events:none;
  }
  .hdr{display:flex; align-items:baseline; gap:10px; margin-bottom:8px}
  .badge{font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:#b6fff6;
         border:1px solid rgba(0,255,245,.35); padding:5px 8px; border-radius:999px}
  .tag{color:var(--muted); font-size:13px}
  .meta{display:grid; grid-template-columns: 1fr; gap:10px; margin:10px 0 0; padding:0}
  .meta > div{display:grid; grid-template-columns: 140px 1fr; gap:12px; align-items:start}
  dt{color:#bcd2ff; font-weight:600}
  dd{margin:0}
  .reasons{margin:0; padding-left:18px}
  @media (prefers-color-scheme: light){
    .cybercard{--bg:#f7fbff; --ink:#0b1530; --muted:#3f4f77}
    .badge{color:#083; border-color:#0dd}
    a{color:#0aa}
  }
</style>

