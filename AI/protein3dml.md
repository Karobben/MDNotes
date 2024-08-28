---
toc: true
url: protein3dml
covercopy: <a href="https://www.nature.com/articles/s41586-023-06415-8">© Joseph L. Watson</a>
priority: 10000
date: 2024-05-29 14:28:45
title: "AI Tools for Protein Structures"
ytitle: "AI Tools for Protein Structures"
description: "AI Tools for Protein Structures"
excerpt: "AI Tools for Protein Structures"
tags: [AI, Machine Learning, 3D, Protein Structure]
category: [Machine Learning, LM, Protein]
cover: "https://imgur.com/LDfRQBk.png"
thumbnail: "https://imgur.com/WhX0s7Q.png"
---


## trRosetta

[^Anishchenko_I_2021]: Anishchenko I, Pellock S J, Chidyausiku T M, et al. De novo protein design by deep network hallucination[J]. Nature, 2021, 600(7889): 547-552.

They inverted this network to generate new protein sequences from scratch, aiming to design proteins with structures and functions not found in nature.By conducting **Monte Carlo sampling** in sequence space and optimizing the predicted structural features, they managed to produce a variety of new protein sequences.

## RFdiffusion

|||
|:-:|:-|
|![RFdiffsion](https://imgur.com/iCXildL.png)|Watson, Joseph L., et al[^Watson_J_2023] published the RFdiffusion at [github](https://github.com/RosettaCommons/RFdiffusion) in 2023. It fine-tune the **RoseTTAFold**[^Baek_M_2021] and designed for tasks like: protein **monomer** design, protein **binder** design, **symmetric oligomer** design, **enzyme active site** scaffolding and symmetric **motif scaffolding** for therapeutic and **metal-binding** protein design. It is a very powerful tool according to the paper. It is based on the Denoising diffusion probabilistic models (**DDPMs**) which is a powerful class of machine learning models demonstrated to generate new photorealistic images in response to text prompts[^Ramesh_A_2021].|


They use the ProteinMPNN[^Dauparas_J_2022] network to subsequently design sequences encoding theses structure. The diffusion model is based on the **DDPMs**. It can not only design a protein from generation, but also able to predict multiple types of interactions as is shown of the left. It was based on the RoseTTAFold.

**Compared with AF2**
- AlphaFold2 is like a very smart detective that can figure out the 3D shape of a protein just by looking at its amino acid sequence. On the other hand, RFdiffusion is more like an architect that designs entirely new proteins with specific properties. Instead of just figuring out shapes, it creates new proteins that can do things like bind to specific molecules or perform certain reactions. This makes it incredibly useful for designing new therapies and industrial enzymes.


[^Watson_J_2023]: Watson J L, Juergens D, Bennett N R, et al. De novo design of protein structure and function with RFdiffusion[J]. Nature, 2023, 620(7976): 1089-1100.
[^Ramesh_A_2021]: Ramesh, A. et al. Zero-shot text-to-image generation. in Proc. 38th International Conference on Machine Learning Vol. 139 (eds Meila, M. & Zhang, T.) 8821–8831 (PMLR, 2021).
[^Baek_M_2021]: Baek M, et al. Accurate prediction of protein structures and interactions using a 3-track network. Science. July 2021.
[^Dauparas_J_2022]: Dauparas J, Anishchenko I, Bennett N, et al. Robust deep learning–based protein sequence design using ProteinMPNN[J]. Science, 2022, 378(6615): 49-56.

<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>
