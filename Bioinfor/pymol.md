---
toc: true
url: pymol
covercopy: <a href="https://pymol.org/2/">© pymol</a>
priority: 10000
date: 2021-09-17 13:57:09
title: "Pymol: the best protein structure viewer"
ytitle: "Pymol"
description: 'Pymol: the best protein structure viewer'
excerpt: "Pymol: the best protein structure viewer"
tags: [Software, Protein, PyMol]
category: [Biology, Bioinformatics, Protein Structure]
cover: "https://pymol.org/2/img/screenshot2.png"
thumbnail: "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/PyMOL_logo.svg/64px-PyMOL_logo.svg.png"
---

## View and Themes

### background color
```bash
bg_color grey30
```

### Display Amino Acid

```bash
set seq_view, on
set seq_view_format, 0
```

## Add something

### Add Hydrogen bonds

Add Hydrogen bonds: [PyMOL tutorial](http://www.protein.osaka-u.ac.jp/rcsfp/supracryst/suzuki/jpxtal/Katsutani/en/hydrogenbond.php)

**Action → find → polar contacts → select from menu**

|![Add Hydrogen bonds](http://www.protein.osaka-u.ac.jp/rcsfp/supracryst/suzuki/jpxtal/Katsutani/figure/hydrogenbond1.png)|
|:-:|
|[&copy; PyMOL tutorial](http://www.protein.osaka-u.ac.jp/rcsfp/supracryst/suzuki/jpxtal/Katsutani/en/hydrogenbond.php)|

## Remove something
cite: [&copy; Jan-Philip Gehrcke; 2011](https://gehrcke.de/2011/06/pymol-remove-hydrogens-and-water/)
```bash
# removing hydrogens
remove (hydro)
remove hydrogens

# removing water
remove resn hoh

# removing solvent
remove solvent
```

## Get informations

### Chain information

```bash
get_chains 4fqi
```

<pre>
 cmd.get_chains:  ['A', 'B', 'C', 'H', 'L']
PyMOL>replace C,4,4
</pre>

### Amino Acid Sequence

source: [pymolwiki](https://pymolwiki.org/index.php/Get_fastastr)

```bash
print(cmd.get_fastastr('all'))

# for only show chain B:
print(cmd.get_fastastr('5WL2 and chain B'))
```

<pre>
>5WL2_H
QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAISWVRQAPGQGLEWMGGIIPIFGTANYAQKFQGRVTI
TADESTSTAYMELSSLRSEDTAVYYCARHGNYYYYSGMDVWGQGTTVTVSSASTKGPSVFPLAPSSKSTS
GGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKP
SNTKVDKRVEPKSCHHHHHH
>5WL2_L
QSVLTQPPSVSEAPRQRVTISCSGSSSNIGNNAVNWYQQLPGKAPKLLIYYDDLLPSGVSDRFSGSKSGT
SASLAISGLQSEDEADYYCAAWDDSLNGAVFGGGTQLTVLGQPKAAPSVTLFPPSSEELQANKATLVCLI
SDFYPGAVTVAWKADSSPVKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTV
APTECS
...
</pre>

## Object manipulation  

### movement
```bash
rotate y,-.2, 4fqh
rotate x,-.2, 4fqh
rotate z,-.2, 4fqh
```

### hide chain

```bash
hide representation [,object]
hide representation [,(selection)]
hide (selection)
```

### Show Chain

```bash
show cartoon, Mos99 and chain A chain B
```

## Colors for Pymol

pymolwiki: [Colors' name and values](https://pymolwiki.org/index.php/color_values)

Assign the name of color into an object
```bash
# assign the color into an object
color red, Mos99
# assign the color into selected molecules 
select part1, Mos99 and chain A and resi 50+56+60
color hotpink, part1
```

Example of set a color by RGB value and assign it into an object
```bash
set_color red2, [1,0.3,0.01]
color red2, Mos99 
```

## Strcture align

|![Protein Structure Align](https://pymolwiki.org/images/thumb/6/6e/After_alignment.png/400px-After_alignment.png)|
|:-:|
|[&copy; Pymol](https://pymolwiki.org/index.php/Align)|


```bash
fetch 1oky 1t46, async=0

# 1) default with outlier rejection
align 1oky, 1t46

# 2) with alignment object, save to clustalw file
align 1oky, 1t46, object=alnobj
save alignment.aln, alnobj

# 3) all-atom RMSD (no outlier rejection) and without superposition
align 1oky, 1t46, cycles=0, transform=0
```

### Partial structure align

Cite: [Queen's University](http://pldserver1.biochem.queensu.ca/~rlc/work/teaching/pymol/alignment/)

```bash
align 5cha and resi 1-100, 2xxl and resi 300-400
# or in short form:
align structure2 & i. 1-100, structure 1 & i. 300-400

# Furthermore, you may wish to restrict the alignment to just the backbone atoms, so you can say:

align structure2 and resi 1-100 and name n+ca+c+o, structure1 and resi 300-400 and name n+ca+c+o

# or in short form:

align structure2 & i. 1-100 & n. n+ca+c+o, structure1 & i. 300-400 & n. n+ca+c+o
```

### Align chains

```bash
align 5cha and chain A+B+C, 2xxl and chain A
```

## Atom

### Atom color

```bash
## change the whole proteins color
color grey90, 2xxl
color grey80, 2xxl 5cha # 2 proteins
```

### Select Atom

[Select Properties](https://pymolwiki.org/index.php/Property_Selectors)

```PyMol PyMol
select aas, resn ASP+GLU in 2xxl
```

Create a variate ass which contain all ASP and GLU residues.




<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>

## Distance

In pymol, you cannot calculate the distance between 2 residues. You can only do it on molecules. For calculate the distance between 2 residues, we could use alpha carbon (CA) as the reference 

```pymol
distance my_distance, 1tzg and chain P and resi 4 and name CA, 1tzg_H_L_P and chain H and resi 50 and name CA
```

For searching the nearest distance between 2 residues, a easiest way is using python pymol module.
But when if you could use python, than, maybe Biopython could be a better choice.

```python
from pymol import cmd

# Load the structures (skip if already loaded)
cmd.load("1tzg.pdb")
cmd.load("1tzg_H_L_P.pdb")

# Select atoms in the specified residues
residue1 = "1tzg and chain P and resi 4"
residue2 = "1tzg_H_L_P and chain H and resi 50"

# Get atom identifiers for both residues
atoms1 = cmd.get_model(residue1).atom
atoms2 = cmd.get_model(residue2).atom

# Iterate over all atom pairs and calculate distances
MIN = 1000
for atom1 in atoms1:
    for atom2 in atoms2:
        distance = cmd.get_distance(f"{residue1} and name {atom1.name}", f"{residue2} and name {atom2.name}")
        print(f"Distance between {atom1.name} in residue {atom1.resi} and {atom2.name} in residue {atom2.resi}: {distance:.2f} Å")
        if distance <= MIN:
            MIN = distance 
```

## Zoom

In pymol, you could zoom into a specific residues. Those different codes works differently. But they could achieve similar effects.

```pymol
zoom 1tzg and chain A and resi 51
center 1tzg and chain A and resi 51
orient 1tzg and chain A and resi 51
```

## Location Moving

### Transition

```python
# Moving the entire chain A 10 unit from 8a95 
cmd.translate([10.0, 0.0, 0.0], '8a95 and chain A')
cmd.rotate('z', 45, chain_to_move)
```

## Change the Chain Name

```python
cmd.alter("Native and chain A", "chain='B'")
```

```pymol
alter Native and chain A, chain='B' 
```
