---
toc: true
url: openMM
covercopy: © Karobben
priority: 10000
date: 2024-12-08 11:48:03
title: "OpenMM, Molecular Dynamic Simulation"
ytitle: "OpenMM, Molecular Dynamic Simulation"
description: "OpenMM, Molecular Dynamic Simulation"
excerpt: ""
tags: []
category: []
cover: ""
thumbnail: ""
---

## Install

More detailed installations: [OpenMM User Guide](http://docs.openmm.org/latest/userguide/application/01_getting_started.html)

```bash
conda create -n openmm python=3.9 -y
conda activate openmm
conda install -c conda-forge openmm
```

**Test the installation**
```bash
python -m openmm.testInstallation
```

<pre>
OpenMM Version: 8.2
Git Revision: 53770948682c40bd460b39830d4e0f0fd3a4b868

There are 4 Platforms available:

1 Reference - Successfully computed forces
2 CPU - Successfully computed forces
3 CUDA - Successfully computed forces
1 warning generated.
1 warning generated.
4 OpenCL - Successfully computed forces

Median difference in forces between platforms:

Reference vs. CPU: 6.29538e-06
Reference vs. CUDA: 6.75176e-06
CPU vs. CUDA: 7.49106e-07
Reference vs. OpenCL: 6.75018e-06
CPU vs. OpenCL: 7.64529e-07
CUDA vs. OpenCL: 1.757e-07

All differences are within tolerance.
</pre>

## Use the GPU in Simulation

```python
from openmm import Platform

# define the platform
platform = Platform.getPlatformByName('CUDA')
properties = {'DeviceIndex': '1', 'Precision': 'mixed'} 

# add the parameter in simulation
simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
```

To be notice: Evene though, you implied that the GPU parameter in the code, it still heavily relies on the CPU.

## PDB fix

Before you run the simulation, you may need to fix the PDB first.

```python
from openmm import app
from openmm.unit import *
from pdbfixer import PDBFixer
from openmm.app import PDBFile


# repair the PDB
def PDB_fix(INPUT, OUPUT):
    fixer = PDBFixer(filename=INPUT)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    #fixer.addSolvent(fixer.topology.getUnitCellDimensions())
    # Remove problematic water molecules and add correct TIP3P water
    fixer.removeHeterogens(keepWater=True)
    PDBFile.writeFile(fixer.topology, fixer.positions, open(OUPUT, 'w'))

PDB_fix('best.pdb', 'best_fix.pdb')
```

## Anchor a protein

==Code was not tested because my protein has hydrophobic surface and it would crush in the water environment== 
```python
# Anchor protein 1 by restraining its atoms
anchor_force = mm.CustomExternalForce('0.5 * k * (x^2 + y^2 + z^2)')
anchor_force.addPerParticleParameter('k')

# Add position restraints to protein 1
for atom in pdb.topology.atoms():
    if atom.residue.chain.id in ['A', 'B']:  # Assume protein 1 is in chain A
        anchor_force.addParticle(atom.index, [1000])  # High force constant

system.addForce(anchor_force)
```

## Pull Protein Force Apply

```python
# Apply a pulling force to protein 2
pulling_force = mm.CustomExternalForce('k_pull * (x - x0)^2')
pulling_force.addPerParticleParameter('k_pull')
pulling_force.addPerParticleParameter('x0')

# Add pulling force to atoms of protein 2 (e.g., chain B)
for atom in pdb.topology.atoms():
    if atom.residue.chain.id in ['B', 'C']:  # Assume protein 2 is in chain B
        pulling_force.addParticle(atom.index, [-1, 1.0])  # Adjust constants

system.addForce(pulling_force)
```

## Change Record

```python
simulation.reporters.append(app.StateDataReporter(
    'best_fix_sld_tr.csv', 10, step=True, potentialEnergy=True, temperature=True))
simulation.reporters.append(app.PDBReporter('best_fix_sld_tr.pdb', 10))
```


## Save the structure as cif


During the simulation, you may wants to add lots of wather molecular. When the number of molecular over than 100,000, pdb format can't handle it anymore. So you want to save it as `cif` format.
```python
with open('best_fix_sld.cif', 'w') as f:
    app.PDBxFile.writeFile(modeller.topology, modeller.positions, f)
```



## Trouble Shot

### No template found for residue 30730 (HOH)

Citation: []()
Error Code:
<pre>
ValueError: No template found for residue 30730 (HOH).  The set of atoms matches HOH, but the bonds are different.
</pre>

Bug reason: [The PDB format doesn't support models with more than 100,000 atoms.](https://github.com/openmm/openmm/issues/3393)

How to solve: save the output as `cif` format by using `PDBxFile`
```diff
- with open('best_fix_sld.pdb', 'w') as f:
+ with open('best_fix_sld.cif', 'w') as f:
-    app.PDBFile.writeFile(modeller.topology, modeller.positions, f)
+    app.PDBxFile.writeFile(modeller.topology, modeller.positions, f)
```


<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>
