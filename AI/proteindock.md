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
1. 1994: Firstly extend the DOCK into DNA-protein Docking and by screening the Cambridge Crystallographic Database, they find that the protein CC-1065 has high score.[^dock_dna]
    - 1999: DREAM++[^dream++]: It is a extent package for Dock. It use Dock to predict binding and evaluated the interaction and predicts the product, finally search to find the prohibits.
2. 2001: **DOCK 4.0**[^dock4]: It added incremental construction (to sample the internal degrees of freedom of the ligand) and random search. In the Dock4, the ligand is not rigid anymore. Ligands with rotatable-bonds generated multiple conformation by other model. 
3. 2006: **DOCK 5.0**[^dock5]:
    - anchoring: new scoring functions, sampling methods and analysis tools; energy minimizing was mentioned during the.
    - scoring: energy scoring function based on the AMBERL: only **intermolecular** van der Waals (VDW) and electrostatic components in the function. 
    - main limitation: Ligands has lots of rotatable-bonds would cause lots of resource. During the test set, ligands with > 7 rotatable bonds were removed.
    - Some test data correction: using "Compute" and "Biopolymer" from **Sybyl**[^Sybyl] to calculate the Gasteiger–Hückel partial electrostatic charges and add hydrogen for residues.
4. 2009: **DOCK 6**[^dock6]: In this version, it extents it's abilities in RNA-ligands. But the rotatable-bonds from the ligands are still limited into 7~13. With the increasing of the RNA, the accuracy are decreased.
    - update scoring in **solvation energy**:
        - Hawkins–Cramer–Truhlar (HCT) generalized Born with solvent-accessible surface area (GB/SA) solvation scoring with optional salt screening
        - Poisson–Boltzmann with solvent-accessible surface area (PB/SA) solvation scoring
        - AMBER molecular mechanics with GB/SA solvation scoring and optional receptor flexibility
    - other scoring:
        - VDW: grid-based form of the Lennard-Jones potential
        - electrostatic: Zap Tool Kit from OpenEye
5. 2013: **DOCK3.7**[^dock3]:


[^dock]: [Kuntz I D, Blaney J M, Oatley S J, et al. A geometric approach to macromolecule-ligand interactions[J]. Journal of molecular biology, 1982, 161(2): 269-288.](https://www.sciencedirect.com/science/article/pii/002228368290153X)
[^dock_dna]: [Grootenhuis P D J, Roe D C, Kollman P A, et al. Finding potential DNA-binding compounds by using molecular shape[J]. Journal of Computer-Aided Molecular Design, 1994, 8: 731-750.](https://link.springer.com/article/10.1007/BF00124018)
[^dream++]: [Makino S, Ewing T J A, Kuntz I D. DREAM++: flexible docking program for virtual combinatorial libraries[J]. Journal of computer-aided molecular design, 1999, 13: 513-532.](https://link.springer.com/article/10.1023/A:1008066310669)
[^dock4]: [Ewing T J A, Makino S, Skillman A G, et al. DOCK 4.0: search strategies for automated molecular docking of flexible molecule databases[J]. Journal of computer-aided molecular design, 2001, 15: 411-428.](https://link.springer.com/article/10.1023/a:1011115820450)
[^dock5]: [Moustakas D T, Lang P T, Pegg S, et al. Development and validation of a modular, extensible docking program: DOCK 5[J]. Journal of computer-aided molecular design, 2006, 20: 601-619.](https://link.springer.com/article/10.1007/s10822-006-9060-4)
[^dock3]: [Coleman R G, Carchia M, Sterling T, et al. Ligand pose and orientational sampling in molecular docking[J]. PloS one, 2013, 8(10): e75992.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0075992)
[^dock6]: [Lang P T, Brozell S R, Mukherjee S, et al. DOCK 6: Combining techniques to model RNA–small molecule complexes[J]. Rna, 2009, 15(6): 1219-1230.](https://rnajournal.cshlp.org/content/15/6/1219.short)
[^Sybyl]: [S. Pérez, C. Meyer, A. Imberty. “Practical tools for accurate modeling of complex carbohydrates and their interactions with proteins” A. Pullman, J. Jortner, B. Pullman (Eds.), Modelling of Biomolecular Structures and Mechanisms, Kluwer Academic Publishers, Dordrecht (1996), pp. 425-454.](https://bcrf.biochem.wisc.edu/all-tutorials/tutorial-materials-guests/185-2/)

|DOCK4|DOCK5|DOCK6|
|:-:|:-:|:-:|
|![DOCK4](https://imgur.com/SJJfgKt.png)|![DOCK5](https://media.springernature.com/full/springer-static/image/art%3A10.1007%2Fs10822-006-9060-4/MediaObjects/10822_2006_9060_Fig1_HTML.gif)|![DOCK6](https://rnajournal.cshlp.org/content/15/6/1219/F1.large.jpg)
||incremental: anchor-and-grow|The number of<br>rotatable-bonds hashuge<br>effects on success rate|

!!! note anchor-and-grow
    The “anchor-and-grow” conformational search algorithm. The algorithm performs the following steps: (1) DOCK perceives the molecule’s rotatable bonds, which it uses to identify an anchor segment and overlapping rigid layer segments. (2) Rigid docking is used to generate multiple poses of the anchor within the receptor. (3) The first layer atoms are added to each anchor pose, and multiple conformations of the layer 1 atoms are generated. An energy score within the context of the receptor is computed for each conformation. (4) The partially grown conformations are ranked by their score and are spatially clustered. The least energetically favorable and spatially diverse conformations are discarded. (5) The next rigid layer is added to each remaining conformation, generating a new set of conformations. (6) Once all layers have been added, the set of completely grown conformations and orientations is returned




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
On the other hand, I didn't find a way to mark the surface residues so we could focus on specific area. Although, GhatGPT said it could do constrained docking, but it seems we could only constrain the range angles of the receptor and the ligand.

|![Hex Dock in SAMSON](https://documentation.samson-connect.net/tutorials/hex/images/hex-results-animation.gif)|
|:-:|
|[© SAMSON](https://documentation.samson-connect.net/tutorials/hex/protein-docking-with-hex/)|

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
