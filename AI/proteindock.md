---
toc: true
url: proteindock
covercopy: <a href='https://dock.compbio.ucsf.edu/'> © UCSF </a>
priority: 10000
date: 2024-10-15 15:51:32
title: "Protein Dock Overview"
ytitle: "Protein Dock Overview"
description: "Protein Dock Tools and algorithm Overview"
excerpt: "Protein Dock Tools and algorithm Overview"
tags: [protein, 3D, dock]
category: [Biology, Bioinformatics, Protein Structure]
cover: "https://imgur.com/lhLMGnt.png"
thumbnail: "https://imgur.com/lhLMGnt.png"
---


## 1982: Dock; Kuntz, Irwin D., et al.[^dock] (Rigid body-shape based)

|![Dock; Kuntz, Irwin D., et al. 1982](https://imgur.com/lsob6Ob.png)|
|:-:|
|© Kuntz, Irwin D., et al. 1982[^dock]|


In this paper, Kuntz present a way of docking prediction by searching the steric overlap based on the knowing surface structure of 2 proteins. It originally developed by Irwin "Tack" Kuntz and colleagues at the University of California, San Francisco (**UCSF**), DOCK was initially used for small-molecule docking. However, it laid the foundation for the development of more advanced docking algorithms and software that could handle macromolecular docking.

In the first generation of the Dock, it focus on 2 rigid bodies. It treat 2 proteins as one object. The goal of this program is to ==fix the 6 degree of freedom (3 transitions and 3 orientations) that determine the best relative position==. For achieving this goal, three rules are followed:
1. No overlap between 2 proteins
2. all hydrogen are pared with N or O within 3.5 Å. 
3. all ligand atoms within the receptor binding cite.



**Dock families:**
- 1994: Firstly extend the DOCK into DNA-protein Docking and by screening the Cambridge Crystallographic Database, they find that the protein CC-1065 has high score.[^dock_dna]
    - 1999: DREAM++[^dream++]: It is a extent package for Dock. It use Dock to predict binding and evaluated the interaction and predicts the product, finally search to find the prohibits.
- 2001: **DOCK 4.0**[^dock4]: It added incremental construction (to sample the internal degrees of freedom of the ligand) and random search. In the Dock4, the ligand is not rigid anymore. Ligands with rotatable-bonds generated multiple conformation by other model. 
- 2006: **DOCK 5.0**[^dock5]:
    - anchoring: new scoring functions, sampling methods and analysis tools; energy minimizing was mentioned during the.
    - scoring: energy scoring function based on the AMBERL: only intermolecular van der Waals (VDW) and electrostatic components in the function. 
- 2009: [**DOCK 6**: Combining techniques to model RNA–small molecule complexes](https://rnajournal.cshlp.org/content/15/6/1219.short)
- 2013: **DOCK3.7**[^dock3]:


[^dock]: [Kuntz I D, Blaney J M, Oatley S J, et al. A geometric approach to macromolecule-ligand interactions[J]. Journal of molecular biology, 1982, 161(2): 269-288.](https://www.sciencedirect.com/science/article/pii/002228368290153X)
[^dock_dna]: [Grootenhuis P D J, Roe D C, Kollman P A, et al. Finding potential DNA-binding compounds by using molecular shape[J]. Journal of Computer-Aided Molecular Design, 1994, 8: 731-750.](https://link.springer.com/article/10.1007/BF00124018)
[^dream++]: [Makino S, Ewing T J A, Kuntz I D. DREAM++: flexible docking program for virtual combinatorial libraries[J]. Journal of computer-aided molecular design, 1999, 13: 513-532.](https://link.springer.com/article/10.1023/A:1008066310669)
[^dock4]: [Ewing T J A, Makino S, Skillman A G, et al. DOCK 4.0: search strategies for automated molecular docking of flexible molecule databases[J]. Journal of computer-aided molecular design, 2001, 15: 411-428.](https://link.springer.com/article/10.1023/a:1011115820450)
[^dock5]: [Moustakas D T, Lang P T, Pegg S, et al. Development and validation of a modular, extensible docking program: DOCK 5[J]. Journal of computer-aided molecular design, 2006, 20: 601-619.](https://link.springer.com/article/10.1007/s10822-006-9060-4)
[^dock3]: [Coleman R G, Carchia M, Sterling T, et al. Ligand pose and orientational sampling in molecular docking[J]. PloS one, 2013, 8(10): e75992.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0075992)

|DOCK4|DOCK5|
|:-:|:-:|
|![](https://imgur.com/SJJfgKt.png)|![](https://media.springernature.com/full/springer-static/image/art%3A10.1007%2Fs10822-006-9060-4/MediaObjects/10822_2006_9060_Fig1_HTML.gif)|


### Compare to Other Related Tools

<div class="c-article-table-container"><div class="c-article-table-border c-table-scroll-wrapper"><div class="c-table-scroll-wrapper__content c-table-scroll-wrapper__fade--transparent" data-component-scroll-wrapper=""><table class="data last-table"><thead class="c-article-table-head"><tr><th class="u-text-left "><p>Method</p></th><th class="u-text-left "><p>Ligand sampling method<sup>a</sup>
                                          </p></th><th class="u-text-left "><p>Receptor sampling method<sup>a</sup>
                                          </p></th><th class="u-text-left "><p>Scoring function<sup>b</sup>
                                          </p></th><th class="u-text-left "><p>Solvation scoring<sup>c,d</sup>
                                          </p></th></tr></thead><tbody><tr><td class="u-text-left "><p>DOCK 4/5 </p></td><td class="u-text-left "><p>IC</p></td><td class="u-text-left "><p>SE</p></td><td class="u-text-left "><p>MM</p></td><td class="u-text-left "><p>DDD, GB, PB</p></td></tr><tr><td class="u-text-left "><p>FlexX/FlexE </p></td><td class="u-text-left "><p>IC</p></td><td class="u-text-left "><p>SE</p></td><td class="u-text-left "><p>ED</p></td><td class="u-text-left "><p>NA</p></td></tr><tr><td class="u-text-left "><p>Glide</p></td><td class="u-text-left "><p>CE&nbsp;+&nbsp;MC</p></td><td class="u-text-left "><p>TS</p></td><td class="u-text-left "><p>MM&nbsp;+&nbsp;ED</p></td><td class="u-text-left "><p>DS</p></td></tr><tr><td class="u-text-left "><p>GOLD </p></td><td class="u-text-left "><p>GA</p></td><td class="u-text-left "><p>GA</p></td><td class="u-text-left "><p>MM&nbsp;+&nbsp;ED</p></td><td class="u-text-left "><p>NA</p></td></tr></tbody></table></div></div><div class="c-article-table-footer"><ol>
                      <li>
                                    <sup>a</sup>Sampling methods are defined as Genetic Algorithm (GA), Conformational Expansion (CE), Monte Carlo (MC), incremental construction (IC), merged target structure ensemble (SE), torsional search (TS)</li>
                      <li>
                                    <sup>b</sup>Scoring functions are defined as either empirically derived (ED) or based on molecule mechanics (MM)</li>
                      <li>
                                    <sup>c</sup>If the package does not accommodate this option, the symbol NA (Not Available) is used</li>
                      <li>
                                    <sup>d</sup>Additional accuracy can be added to the scoring function using implicit solvent models. The most commonly used options are distance dependent dielectric (DDD), a parameterized desolvation term (DS), generalized Born (GB) and linearized Poisson Boltzmann (PB)</li>
                    </ol></div></div>



### 2003: ZDock

ZDock family
- 2003: **ZDOCK**: An initial-stage protein-docking algorithm
- 2005: **M-ZDOCK**: a grid-based approach for Cn symmetric multimer docking

### 2004: ClusPro

[ClusPro: a fully automated algorithm for protein–protein docking](https://academic.oup.com/nar/article/32/suppl_2/W96/1040440)


### 2010: Hex

[Ultra-fast FFT protein docking on graphics processors](https://academic.oup.com/bioinformatics/article/26/19/2398/229220)

[Home page](https://hex.loria.fr/), [Documentation](https://hex.loria.fr/manual800/hex_manual.html)

Hex is extremely fast but lack of accuracy. I tried to sampling over 100,1000 but results even close to native structure.
On the other hand, I didn't find a way to mark the surface residues so we could focus on specific area.


### 2014: rDock

[rDock: a fast, versatile and open source program for docking ligands to proteins and nucleic acids](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003571)


### 2018: InterEvDock

[Protein-Protein Docking Using Evolutionary Information](https://link.springer.com/protocol/10.1007/978-1-4939-7759-8_28)








<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>
