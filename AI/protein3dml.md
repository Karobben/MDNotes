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

## AlphaFold2

||MSA is important|
|:-:|:-|
|![](https://imgur.com/yaOCXym.png)|How does the Multiple Sequence Alignment (MSA) help: the sequences alignment could help to encode the residues contact map. When a residues mutated, the contact residues are likely to be mutated, too. This kind of features could be captured by models.|

## ESM2

ESM, or **Evolutionary Scale Modeling**, is a family of protein language models developed by Meta AI. ESM2 is the second generation of this model, which has been trained on a larger dataset and incorporates several architectural improvements over its predecessor, ESM-1b.

### How it was trained

ESM-2 is trained to predict the identity of amino acids that have been randomly masked out of protein sequences. This is similar to how models like BERT are trained for natural language processing tasks. The model learns to understand the context of a protein sequence by predicting the masked amino acids based on the surrounding sequence.

Transformer models that are trained with masked language modeling are known to develop attention patterns that correspond to the ==residue-residue contact map== of the protein


ESMFold is a fully end-to-end single-sequence structure predictor, by training a folding head for ESM-2

|![](https://www.science.org/cms/10.1126/science.ade2574/asset/210b61b0-bbba-4a17-bc33-4dabd5c3e49b/assets/images/large/science.ade2574-f1.jpg)|
|:-:|
|![](https://www.science.org/cms/10.1126/science.ade2574/asset/66f4635b-54eb-498b-8049-8a816a5d9c8b/assets/images/large/science.ade2574-f2.jpg)|
|[© Zeming Lin](https://www.science.org/doi/10.1126/science.ade2574)|

## trRosetta

transform-restrained Rosetta

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

- **Residues encoding**: The sequences are encoded using a one-hot encoding scheme which make the model more efficient and way faster than other methods. The orientation of the residues are also encoded.
- There are 8 block to update the encoding features sequencially. Unly like the AF, the weight are difference and not shared.
- **Training data set**: 7084 structures from SAbDab. Filtering: No missing residues and resolution ≤ 3.5 Å 
- **Architect**: The architecture of the deep learning model behind ABodyBuilder2 is an antibody-specific version of the structure module in **AlphaFold-Multimer** with several modifications
- **Frame Aligned Point Error (FAPE) loss** (like AFM)
- **Model selection**: Unlike the AF2, ABodyBuilder2 doesn't rank and select the best structure. It calculate the similarities and select the one most closing to the everage structure. They mentioned that it could "reduces the method’s sensitivity to small fluctuations in the training set. It also results in a small improvement in prediction accuracy." 

A set of deep learning models trained to accurately predict the structure of antibodies (ABodyBuilder2), nanobodies (NanoBodyBuilder2) and T-Cell receptors (TCRBuilder2). ImmuneBuilder generates structures with state of the art accuracy while being much faster than AlphaFold2.

Experience it online: [Google Colab](https://colab.research.google.com/github/brennanaba/ImmuneBuilder/blob/main/notebook/ImmuneBuilder.ipynb) 
GitHub: [oxpig/ImmuneBuilder](https://github.com/oxpig/ImmuneBuilder)

They have built three models
- **ABodyBuilder2**, an antibody-specific model
- **NanoBodyBuilder2**, a nanobody-specific model
- **TCRBuilder2**, a TCR-specific model.

How: compare 34 antibody structures recently added

<table class="data last-table"><thead class="c-article-table-head"><tr><th class="u-text-left "><p>Method</p></th><th class="u-text-left "><p>CDR-H1</p></th><th class="u-text-left "><p>CDR-H2</p></th><th class="u-text-left "><p>CDR-H3</p></th><th class="u-text-left "><p>Fw-H</p></th><th class="u-text-left "><p>CDR-L1</p></th><th class="u-text-left "><p>CDR-L2</p></th><th class="u-text-left "><p>CDR-L3</p></th><th class="u-text-left "><p>Fw-L</p></th></tr></thead><tbody><tr><td class="u-text-left "><p>ABodyBuilder (ABB)</p></td><td class="u-text-left "><p>1.53</p></td><td class="u-text-left "><p>1.09</p></td><td class="u-text-left "><p>3.46</p></td><td class="u-text-left "><p>0.65</p></td><td class="u-text-left "><p>0.71</p></td><td class="u-text-left "><p>0.55</p></td><td class="u-text-left "><p>1.18</p></td><td class="u-text-left "><p>0.59</p></td></tr><tr><td class="u-text-left "><p>ABlooper (ABL)</p></td><td class="u-text-left "><p>1.18</p></td><td class="u-text-left "><p>0.96</p></td><td class="u-text-left "><p>3.34</p></td><td class="u-text-left "><p>0.63</p></td><td class="u-text-left "><p>0.78</p></td><td class="u-text-left "><p>0.63</p></td><td class="u-text-left "><p>1.08</p></td><td class="u-text-left "><p>0.61</p></td></tr><tr><td class="u-text-left "><p>IgFold (IgF)</p></td><td class="u-text-left "><p>0.86</p></td><td class="u-text-left "><p>0.77</p></td><td class="u-text-left "><p>3.28</p></td><td class="u-text-left "><p>0.58</p></td><td class="u-text-left "><p>0.55</p></td><td class="u-text-left "><p>0.43</p></td><td class="u-text-left "><p>1.12</p></td><td class="u-text-left "><p>0.60</p></td></tr><tr><td class="u-text-left "><p>EquiFold (EqF)</p></td><td class="u-text-left "><p>0.86</p></td><td class="u-text-left "><p>0.80</p></td><td class="u-text-left "><p>3.29</p></td><td class="u-text-left "><p>0.56</p></td><td class="u-text-left "><p>0.47</p></td><td class="u-text-left "><p>0.41</p></td><td class="u-text-left "><p>0.93</p></td><td class="u-text-left "><p><b>0.54</b></p></td></tr><tr><td class="u-text-left "><p>AlphaFold-M (AFM)</p></td><td class="u-text-left "><p>0.86</p></td><td class="u-text-left "><p><b>0.68</b></p></td><td class="u-text-left "><p>2.90</p></td><td class="u-text-left "><p>0.55</p></td><td class="u-text-left "><p>0.47</p></td><td class="u-text-left "><p><b>0.40</b></p></td><td class="u-text-left "><p><b>0.83</b></p></td><td class="u-text-left "><p><b>0.54</b></p></td></tr><tr><td class="u-text-left "><p>ABodyBuilder2 (AB2)</p></td><td class="u-text-left "><p><b>0.85</b></p></td><td class="u-text-left "><p>0.78</p></td><td class="u-text-left "><p><b>2.81</b></p></td><td class="u-text-left "><p><b>0.54</b></p></td><td class="u-text-left "><p><b>0.46</b></p></td><td class="u-text-left "><p>0.44</p></td><td class="u-text-left "><p>0.87</p></td><td class="u-text-left "><p>0.57</p></td></tr></tbody></table>

!!! Note What is an acceptable RMSD?
    The experimental error in protein structures generated via X-ray crystallography has been estimated to be around **0.6Å** for regions with organised secondary structures (such as the antibody frameworks) and around **1Å** for protein loops.[^Eyal_E]

### Side Chain Prediction

ABlooper and IgFold only predict the position of backbones, leaving the side chain to OpenMM[^OpenMM] and Rosetta[^Rosetta], while EquiFold, AlphaFold-Multimer and ABodyBuilder2, all of which output all-atom structures.
## ABodyBuilder3

![ABodyBuilder3 Architecture](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/bioinformatics/40/10/10.1093_bioinformatics_btae576/2/btae576f1.jpeg?Expires=1747325501&Signature=BqZJxmhW3O6dAYiXk69n0UHAovXibmMDl4HMEr1hXLD7gxOEGlglePorvZeBQqPiNi~SDiSbeZJMfVxAnwYKGoVSGtE26zDJPK1g97kHJ48kqczwxUusgUAuksw8h6DnnwMAxnm3Fg1qukiQ6fkumNrKnHUIahs5oJuVX09ZbwvsfeIlgk82LKW9oEvg6rGSgI6jgWl7CYg2PtITQL5KW6pJ0cVOOxwl2axXWoc2Xk5V4GLYAPp7euuS9NedRV8KXkZipoGoj3caMcxitmkelgxfzAedkLYPAZ-cL-CDF4S0Y-E6i2EKLPO347gAA3gMWzYuh2utbOxXYl7ExkimNQ__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)

Like the ABodyBuilder2, the ABodyBuilder3 is comming from the **Deane** Lab at **University of Oxford**[^Kenlay_24]. They updated this model in 2024 by bring some new features and slightly improved the performance.

[^Kenlay_24]: Kenlay H, Dreyer F A, Cutting D, et al. ABodyBuilder3: improved and scalable antibody structure predictions[J]. Bioinformatics, 2024, 40(10): btae576.


Upgrading:
- In this time, they removed super-long CDRH3 (>=30) antibodies from Bovine.
- OpenMM[^OpenMM] and YASARA[^YASARA] were test for structure refinement. They found that **YASARA is better**. (By checking the result from [Supplemental](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/bioinformatics/40/10/10.1093_bioinformatics_btae576/2/btae576_supplementary_data.pdf?Expires=1746002435&Signature=NWXox9Xwa3JSGup5C368J9F7bWI11z7ww9KKOac8GbrhGxaI499FI1zh5SHnxpLRLJXk9nrAupyYB-LmceuQMQcPHg3Gh1JiUBA1Kl7lQ73XVlGwIMYxcOjKIdS873RIZwjxN3yPr6EjLEix9vUtewTX4FHIH0A2--ujEuWbl5ghddS~-vnuZ~sp54oB3QooxFzfwSmXPm6nmpeY~lz~wjrtoPoWK-jk~0KEz3Z~8sZ75uEd-aSRONXns-bDKEXAwS04a5DnRWEEViJtuZJH5dfYm0bfy~E2oH4M7cV47yC3ZoYI4cSOQdJCN22959dumMzf8JMRB1M-SS0s1XMDmg__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA), the results are pretty much the same) 
- Language Embedding. In ABodyBuilder3, they now given the options of LM embedding the sequences. As it's show under the table below, the performance is slightly improved in CDRH3 region. They used the ProtT5[^ProtT5] model and embedding the H chain and L chain separately. They also tried paired IgT5 and IgBert[^IgBert] from the same lab and found the general model is better.

[^IgBert]: Kenlay H, Dreyer F A, Kovaltsuk A, et al. Large scale paired antibody language models[J]. PLOS Computational Biology, 2024, 20(12): e1012646. 
[^YASARA]: Krieger E, Vriend G. New ways to boost molecular dynamics simulations. J Comput Chem 2015;36:996–1007.
[^ProtT5]: Elnaggar A, Heinzinger M, Dallago C et al. Prottrans: toward understanding the language of life through self-supervised learning. IEEE Trans Pattern Anal Mach Intell 2021;44:7112–27.

<table role="table" aria-labelledby="
                        label-13673" aria-describedby="
                        caption-13673"><thead><tr><th><span aria-hidden="true" style="display: none;">
            .&nbsp;</span></th><th>CDRH1<span aria-hidden="true" style="display: none;">
            .&nbsp;</span></th><th>CDRH2<span aria-hidden="true" style="display: none;">
            .&nbsp;</span></th><th>CDRH3<span aria-hidden="true" style="display: none;">
            .&nbsp;</span></th><th>Fw-H<span aria-hidden="true" style="display: none;">
            .&nbsp;</span></th><th>CDRL1<span aria-hidden="true" style="display: none;">
            .&nbsp;</span></th><th>CDRL2<span aria-hidden="true" style="display: none;">
            .&nbsp;</span></th><th>CDRL3<span aria-hidden="true" style="display: none;">
            .&nbsp;</span></th><th>Fw-L<span aria-hidden="true" style="display: none;">
            .&nbsp;</span></th></tr></thead><tbody><tr><td>ABodyBuilder2</td><td>0.41</td><td>0.38</td><td>0.57</td><td>0.50</td><td>0.47</td><td>0.48</td><td>0.72</td><td>0.40</td></tr><tr><td>ABodyBuilder3</td><td>0.58</td><td>0.26</td><td>0.61</td><td>0.48</td><td>0.60</td><td>0.20</td><td>0.68</td><td>0.67</td></tr><tr><td>ABodyBuilder3-LM</td><td>0.69</td><td>0.36</td><td>0.73</td><td>0.39</td><td>0.72</td><td>0.52</td><td>0.68</td><td>0.58</td></tr></tbody></table>


## Equifold

Designing proteins to achieve specific functions often requires in silico modeling of their properties at high throughput scale and can significantly benefit from fast and accurate protein structure prediction. We introduce EquiFold, a new end-to-end differentiable, SE(3)-equivariant, all-atom protein structure prediction model. EquiFold uses a novel coarse-grained representation of protein structures that does not require multiple sequence alignments or protein language model embeddings, inputs that are commonly used in other state-of-the-art structure prediction models. Our method relies on geometrical structure representation and is substantially smaller than prior state-of-the-art models. In preliminary studies, EquiFold achieved comparable accuracy to AlphaFold but was orders of magnitude faster. The combination of high speed and accuracy make EquiFold suitable for a number of downstream tasks, including protein property prediction and design.

https://github.com/Genentech/equifold

## IgFold

![IgFold architecture](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41467-023-38063-x/MediaObjects/41467_2023_38063_Fig1_HTML.png?as=webp)

IgFold[^IgFold] was developed by the **Jeffrey Gray** Lab at **Johns Hopkins University**. It is a fine-tuned AlphaFold-Multimer model. But instead of doing multiple sequences alignments, it embedding the residues like the esm. They trained a large dataset of natural antibodies based on Birt, which is AntiBERTy[^AntiBERTy] and trained on 500 million antibody sequences from the SAbDab database. In the folding module, it using full connected residues network, an type of optimized GNN. As you can see the graph from the AntiBERTy below, the LM can do very well on CDRH1 and CDRH2 but not very good on CDRH3,  With the help of the AbtiBERTy, it can predict the structures much faster. For the structure prediction training, not only 4K unique antibodies structures from PDB they used, but also about 16K of **AF2 predicted structures** (pLDDT >=85) are used[^Jeffery_rugster].

![IgFold, Folding Module](https://imgur.com/3cL04oe.png)
![IgFold; Supplemental](https://imgur.com/1n7Sdlv.png)

|![IgLM](https://imgur.com/vfIJFSh.png)|
|:-|
| AntiBERTy was lay in the bioRxiv for many years. Then time when they finally published it, they iterated the model to IgLM[^Shuai_2023]. Instead of the mask model, it became like an RNN model|

[^AntiBERTy]: Ruffolo J A, Gray J J, Sulam J. Deciphering antibody affinity maturation with language models and weakly supervised learning[J]. arXiv preprint arXiv:2112.07782, 2021.
[^IgFold]:[Ruffolo J A, Chu L S, Mahajan S P, et al. Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies[J]. Nature communications, 2023, 14(1): 2389.](https://www.nature.com/articles/s41467-023-38063-x)
[^Shuai_2023]: [Shuai R W, Ruffolo J A, Gray J J. IgLM: Infilling language modeling for antibody sequence design[J]. Cell Systems, 2023, 14(11): 979-989. e4.](https://www.cell.com/cell-systems/fulltext/S2405-4712(23)00271-5?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2405471223002715%3Fshowall%3Dtrue#fig1)
[^Jeffery_rugster]:[Jeffrey Gray: Artificial Intelligence Tools for Antibody Engineering and Protein Docking; 2024; YouTube](https://www.youtube.com/watch?v=oSTHQQYoGQs)


!!! note Why they including AF2 predicted structures in training set?  
    because an old machine-learning professor told Jeffery that it would be helpful even thought they are not very accurate. And at the end, the performance of the IgFold works slightly better on the AF2. In the end of the Jeffery's talk, they said that they observed that with out adding AF2 predicted data, the performance is not as good as now. (They only took the structure with good confidence, not all data)

Official repository for IgFold: Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies.

The code and pre-trained models from this work are made available for non-commercial use (including at commercial entities) under the terms of the JHU Academic Software License Agreement. For commercial inquiries, please contact Johns Hopkins Tech Ventures at awichma2@jhu.edu.


Try antibody structure prediction in Google Colab: https://github.com/Graylab/IgFold

!!! note Note Personal experience
    IgFold is much slower than ABodyBuilder2. I think it could because IgFold is using AntiBERTy to embedding the sequences. But the ABodyBuilder2 is using one-hot encoding.
    I feel that the IgFold is kind of horrible in CDRH3 regions. It predicted CDRH3 loop in an erect conformation incorrectly. It is worse than ABodyBuilder2. It only takes the perfect Fab sequences. Any longer seqeunces would end up as a mass.




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
