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

## ImmuneBuilder

[ImmuneBuilder: Deep-Learning models for predicting the structures of immune proteins](https://www.nature.com/articles/s42003-023-04927-7)


### Method of ABodyBuilder2

> -  First, the heavy and light chain sequences are fed into four separate deep-learning models to predict an ensemble of structures. The closest structure to the average is then selected and refined using OpenMM[^OpenMM] to remove clashes and other stereo-chemical errors. The same pipeline is used for NanoBodyBuilder2 and TCRBuilder2.
> - **Training data set**: 7084 structures from SAbDab. Filtering: No missing residues and resolution ≤ 3.5 Å 
> - **Architect**: The architecture of the deep learning model behind ABodyBuilder2 is an antibody-specific version of the structure module in **AlphaFold-Multimer** with several modifications
> - **Frame Aligned Point Error (FAPE) loss** (like AFM)

A set of deep learning models trained to accurately predict the structure of antibodies (ABodyBuilder2), nanobodies (NanoBodyBuilder2) and T-Cell receptors (TCRBuilder2). ImmuneBuilder generates structures with state of the art accuracy while being much faster than AlphaFold2.

Experience it online: [Google Colab](https://colab.research.google.com/github/brennanaba/ImmuneBuilder/blob/main/notebook/ImmuneBuilder.ipynb) 
GitHub: [oxpig/ImmuneBuilder](https://github.com/oxpig/ImmuneBuilder)

They have built three models
- **ABodyBuilder2**, an antibody-specific model
- **NanoBodyBuilder2**, a nanobody-specific model
- **TCRBuilder2**, a TCR-specific model.

It compared the performance with other similar tools:
- homology modelling method; **ABodyBuilder**[^ABodyBuilder] 
- general protein structure prediction method: **AlphaFold-Multimer**[^AFM]
- antibody-specific methods: ABlooper[^ABL] (ABL), IgFold[^IgF] (IgF) and EquiFold[^EqF] (EqF)

How: compare 34 antibody structures recently added

<table class="data last-table"><thead class="c-article-table-head"><tr><th class="u-text-left "><p>Method</p></th><th class="u-text-left "><p>CDR-H1</p></th><th class="u-text-left "><p>CDR-H2</p></th><th class="u-text-left "><p>CDR-H3</p></th><th class="u-text-left "><p>Fw-H</p></th><th class="u-text-left "><p>CDR-L1</p></th><th class="u-text-left "><p>CDR-L2</p></th><th class="u-text-left "><p>CDR-L3</p></th><th class="u-text-left "><p>Fw-L</p></th></tr></thead><tbody><tr><td class="u-text-left "><p>ABodyBuilder (ABB)</p></td><td class="u-text-left "><p>1.53</p></td><td class="u-text-left "><p>1.09</p></td><td class="u-text-left "><p>3.46</p></td><td class="u-text-left "><p>0.65</p></td><td class="u-text-left "><p>0.71</p></td><td class="u-text-left "><p>0.55</p></td><td class="u-text-left "><p>1.18</p></td><td class="u-text-left "><p>0.59</p></td></tr><tr><td class="u-text-left "><p>ABlooper (ABL)</p></td><td class="u-text-left "><p>1.18</p></td><td class="u-text-left "><p>0.96</p></td><td class="u-text-left "><p>3.34</p></td><td class="u-text-left "><p>0.63</p></td><td class="u-text-left "><p>0.78</p></td><td class="u-text-left "><p>0.63</p></td><td class="u-text-left "><p>1.08</p></td><td class="u-text-left "><p>0.61</p></td></tr><tr><td class="u-text-left "><p>IgFold (IgF)</p></td><td class="u-text-left "><p>0.86</p></td><td class="u-text-left "><p>0.77</p></td><td class="u-text-left "><p>3.28</p></td><td class="u-text-left "><p>0.58</p></td><td class="u-text-left "><p>0.55</p></td><td class="u-text-left "><p>0.43</p></td><td class="u-text-left "><p>1.12</p></td><td class="u-text-left "><p>0.60</p></td></tr><tr><td class="u-text-left "><p>EquiFold (EqF)</p></td><td class="u-text-left "><p>0.86</p></td><td class="u-text-left "><p>0.80</p></td><td class="u-text-left "><p>3.29</p></td><td class="u-text-left "><p>0.56</p></td><td class="u-text-left "><p>0.47</p></td><td class="u-text-left "><p>0.41</p></td><td class="u-text-left "><p>0.93</p></td><td class="u-text-left "><p><b>0.54</b></p></td></tr><tr><td class="u-text-left "><p>AlphaFold-M (AFM)</p></td><td class="u-text-left "><p>0.86</p></td><td class="u-text-left "><p><b>0.68</b></p></td><td class="u-text-left "><p>2.90</p></td><td class="u-text-left "><p>0.55</p></td><td class="u-text-left "><p>0.47</p></td><td class="u-text-left "><p><b>0.40</b></p></td><td class="u-text-left "><p><b>0.83</b></p></td><td class="u-text-left "><p><b>0.54</b></p></td></tr><tr><td class="u-text-left "><p>ABodyBuilder2 (AB2)</p></td><td class="u-text-left "><p><b>0.85</b></p></td><td class="u-text-left "><p>0.78</p></td><td class="u-text-left "><p><b>2.81</b></p></td><td class="u-text-left "><p><b>0.54</b></p></td><td class="u-text-left "><p><b>0.46</b></p></td><td class="u-text-left "><p>0.44</p></td><td class="u-text-left "><p>0.87</p></td><td class="u-text-left "><p>0.57</p></td></tr></tbody></table>

- What is an acceptable RMSD[^Eyal_E]?

!!! Note What is an acceptable RMSD?
    The experimental error in protein structures generated via X-ray crystallography has been estimated to be around **0.6Å** for regions with organised secondary structures (such as the antibody frameworks) and around **1Å** for protein loops.


### Side Chain Prediction

ABlooper and IgFold only predict the position of backbones, leaving the side chain to OpenMM[^OpenMM] and Rosetta[^Rosetta], while EquiFold, AlphaFold-Multimer and ABodyBuilder2, all of which output all-atom structures.

## equifold

Designing proteins to achieve specific functions often requires in silico modeling of their properties at high throughput scale and can significantly benefit from fast and accurate protein structure prediction. We introduce EquiFold, a new end-to-end differentiable, SE(3)-equivariant, all-atom protein structure prediction model. EquiFold uses a novel coarse-grained representation of protein structures that does not require multiple sequence alignments or protein language model embeddings, inputs that are commonly used in other state-of-the-art structure prediction models. Our method relies on geometrical structure representation and is substantially smaller than prior state-of-the-art models. In preliminary studies, EquiFold achieved comparable accuracy to AlphaFold but was orders of magnitude faster. The combination of high speed and accuracy make EquiFold suitable for a number of downstream tasks, including protein property prediction and design.

https://github.com/Genentech/equifold

## IgFold


Official repository for IgFold: Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies.

The code and pre-trained models from this work are made available for non-commercial use (including at commercial entities) under the terms of the JHU Academic Software License Agreement. For commercial inquiries, please contact Johns Hopkins Tech Ventures at awichma2@jhu.edu.

Try antibody structure prediction in Google Colab.

https://github.com/Graylab/IgFold

!!! Personal experience
    I feel that the IgFold is kind of horrible in CDRH3 regions. It predicted CDRH3 loop in an erect conformation incorrectly. It is worse than ABodyBuilder2. It is even slower than ABodyBuilder2, too. It only takes the perfect Fab sequences. Any longer seqeunces would end up as a mass.




[^ABL]: Abanades, B., Georges, G., Bujotzek, A. & Deane, C. M. ABlooper: fast accurate antibody CDR loop structure prediction with accuracy estimation. Bioinformatics 38, 1877–1880 (2022).
[^ABodyBuilder]: Leem, J., Dunbar, J., Georges, G., Shi, J. & Deane, C. M. ABodyBuilder: automated antibody structure prediction with data-driven accuracy estimation. MAbs 8, 1259–1268 (2016).
[^AFM]: Evans, R. et al. Protein complex prediction with AlphaFold-Multimer. bioRxiv (2021).
[^EqF]: Lee, J. H. et al. Equifold: Protein structure prediction with a novel coarse-grained structure representation. bioRxiv (2022).
[^Eyal_E]: Eyal, E., Gerzon, S., Potapov, V., Edelman, M. & Sobolev, V. The limit of accuracy of protein modeling: influence of crystal packing on protein structure. J. Mol. Biol. 351, 431–442 (2005).Return to ref 35 in article
[^IgF]: Ruffolo, J. A., Chu, L.-S., Mahajan, S. P. & Gray, J. J. Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies. Nat. Commun. 14, 2389 (2023).
[^OpenMM]: Eastman, P. et al. OpenMM 7: rapid development of high-performance algorithms for molecular dynamics. PLoS Comput. Biol. 13, e1005659 (2017).
[^Rosetta]: Alford, R. F. et al. The Rosetta all-atom energy function for macromolecular modeling and design. J. Chem. Theory Comput. 13, 3031–3048 (2017).


<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>
