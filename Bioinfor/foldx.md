---
toc: true
url: foldx
covercopy: © Karobben
priority: 10000
date: 2024-07-06 17:11:38
title: "FoldX"
ytitle: "FoldX"
description: "FoldX"
excerpt: "FoldX"
tags: [Bioinformatics, Protein Structure]
category: [Biology, Bioinformatics, Software ]
cover: "https://foldxsuite.crg.eu/sites/default/files/FoldX.png"
thumbnail: "https://foldxsuite.crg.eu/sites/default/files/FoldX.png"
---


## Run the FoldX

In this example, I am using the **7ekb** as example

```bash
# download the pdb
wget https://files.rcsb.org/view/7ekb.pdb
# Repair the PDB. After repaired it, you'll get the 7ekb_Repair.pdb for the next step
FoldX --command=RepairPDB --pdb=7ekb.pdb
# calculate the free energy of the PDB  
FoldX --command=Stability  --pdb=7ekb_Repair.pdb
```

## RepairPDB

### Why Repair PDB?

According to ChatGPT4o, `RepairPDB` command in FoldX is a crucial step to ensure the quality and integrity of your PDB file before performing stability calculations or other analyses. Also, you could found more information from [document](https://foldxsuite.crg.eu/command/RepairPDB)


1. **Fix Structural Issues**:
   - **Correcting Errors**: PDB files obtained from experiments like X-ray crystallography or cryo-EM often have missing atoms, residues, or other structural issues that can affect downstream analyses. `RepairPDB` fixes these issues to ensure a complete and accurate structure.
   - **Adding Missing Atoms**: The command can add missing atoms, such as hydrogen atoms, which are essential for energy calculations.

2. **Standardizing the Structure**:
   - **Normalization**: `RepairPDB` standardizes the structure to ensure that all residues and atoms are in the correct format and positions. This includes correcting bond lengths and angles to standard values.
   - **Removing Non-standard Residues**: It can remove or correct non-standard residues and ligands that might interfere with calculations.

3. **Improving Energy Calculations**:
   - **Optimizing Geometry**: The command optimizes the geometry of the protein, ensuring that the atomic positions are energetically favorable. This leads to more accurate stability and free energy calculations.
   - **Minimizing Steric Clashes**: It identifies and resolves steric clashes (where atoms are too close to each other), which can distort energy calculations.

4. **Ensuring Compatibility**:
   - **Consistency**: Running `RepairPDB` ensures that your PDB file is compatible with FoldX’s algorithms, reducing the risk of errors during subsequent steps.


!!! question How does the Output looks like?

<pre>
Residue LYSH222 has high Energy, we mutate it to itself
Repair Residue ID= LYSH222

BackHbond       =               -317.22
SideHbond       =               -137.87
Energy_VdW      =               -476.42
Electro         =               -15.23
Energy_SolvP    =               628.80
Energy_SolvH    =               -624.90
Energy_vdwclash =               15.60
energy_torsion  =               9.33
backbone_vdwclash=              143.43
Entropy_sidec   =               245.04
Entropy_mainc   =               632.72
water bonds     =               0.00
helix dipole    =               -0.35
loop_entropy    =               0.00
cis_bond        =               4.50
disulfide       =               -13.95
kn electrostatic=               -0.25
partial covalent interactions = 0.00
Energy_Ionisation =             1.07
Entropy Complex =               0.00
-----------------------------------------------------------
Total          = 				  -49.10
</pre>

It took me ==2m 48s==. It only work in single thread and cannot move on multiple threads. I guess because it works by following the order of the AA and the `Total` is depending on the previous values. So, it can't work on multiple threads. 

Here is the result of before and after repairing. The RMS=0.01 which means it almost the same. But the slightly different are mainly focus on the loos area. In the picture present below, the left panel with green structure is the raw pdb file from PDB database. The light blue structure on the right is the corrected by FoldX. Red structure is antigen. As I marked on the left panel, 2 beta-sheets and 1 alpha helix are deleted and become random loop. Those area from the antibody are very closing to the antigen. So, technically, random loop would make more sense to me.

![](https://imgur.com/i8aPrTy.png)


## Stability Calculations

After repaired the PDB file, you can get the result immediately. 

<pre>
   ********************************************
   ***                                      ***
   ***             FoldX 5.1 (c)            ***
   ***                                      ***
   ***     code by the FoldX Consortium     ***
   ***                                      ***
   ***     Jesper Borg, Frederic Rousseau   ***
   ***    Joost Schymkowitz, Luis Serrano   ***
   ***    Peter Vanhee, Erik Verschueren    ***
   ***     Lies Baeten, Javier Delgado      ***
   ***       and Francois Stricher          ***
   *** and any other of the 9! permutations ***
   ***   based on an original concept by    ***
   ***   Raphael Guerois and Luis Serrano   ***
   ********************************************

Stability >>>

1 models read: 7ekb_Repair.pdb

BackHbond       =               -332.04
SideHbond       =               -163.29
Energy_VdW      =               -481.14
Electro         =               -17.42
Energy_SolvP    =               626.91
Energy_SolvH    =               -633.28
Energy_vdwclash =               13.20
energy_torsion  =               9.65
backbone_vdwclash=              144.57
Entropy_sidec   =               259.27
Entropy_mainc   =               634.24
water bonds     =               0.00
helix dipole    =               -0.40
loop_entropy    =               0.00
cis_bond        =               4.50
disulfide       =               -13.95
kn electrostatic=               -0.41
partial covalent interactions = 0.00
Energy_Ionisation =             1.14
Entropy Complex =               0.00
-----------------------------------------------------------
Total          = 				  -93.01

FINISHING STABILITY ANALYSIS OPTION

Your file run OK
End time of FoldX: Sat Jul  6 17:23:18 2024
Total time spend: 0.85 seconds.
</pre>

<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>
