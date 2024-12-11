---
toc: true
url: openMM
covercopy: © Karobben
priority: 10000
date: 2024-12-08 11:48:03
title: "OpenMM, Molecular Dynamic Simulation"
ytitle: "OpenMM, Molecular Dynamic Simulation"
description: "OpenMM, Molecular Dynamic Simulation"
excerpt: "123"
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

## Save the structure from the simulation

```python
# After finishing your MD steps:
state = simulation.context.getState(getPositions=True)
positions = state.getPositions()

# Write out the final structure to a PDB file
with open('final_structure.pdb', 'w') as f:
    PDBFile.writeFile(simulation.topology, positions, f)
```

## In Action: Protein in Water

|||
|:-:|:-|
|![](https://imgur.com/JnRSG6l.gif)|With the help of openMM toolkit, you can simulate the protein in water easily. Here is the code from the [document](https://openmm.github.io/openmm-cookbook/latest/notebooks/tutorials/protein_in_water.html).|


In the simulation, the main codes is explained:
1. read the pdb file with `PDBFile`
2. Specify the force filed. 
3. Clean water and add water as a period box. In this step, you can add the water in the force filed based on the size of the filed. Our you can test small filed. The filed is a period box (replicated infinitely in all directions) which means when the object moves to the end of one side, it will not run out of the box, but coming back from **against face**.
4. Setup the integrator:
    - `forcefield.createSystem`: 
        - `modeller.topology`: you'll add you molecular (protein)
        - `nonbondedMethod=PME`: <font title='ChatGPT o1' color=gray>specifies how long-range electrostatic interactions. Simply cutting them off at a certain radius can introduce errors. **PME** (Particle Mesh Ewald) uses a combination of direct space calculations (for short distances) and reciprocal space calculations (using fast Fourier transforms) to accurately handle these interactions.</font>
        - `nonbondedCutoff=1.0*nanometer`: <font title='ChatGPT o1' color=gray>When using a cutoff-based approach (like nonbondedCutoff=1.0*nanometer), the simulation engine directly calculates the vdW (Lennard-Jones) interactions only between pairs of atoms that are within that 1 nm cutoff distance. If two atoms are farther apart than 1 nm, their vdW interactions are not explicitly computed.</font>
    - `integrator`: This is the place to given the value of the Tm and time scale.
5. `simulation.minimizeEnergy()`: Start to minimize local Energy.
6. Setup report: set up the report to record the energy change and save the trajectory.
7. Simulate in the ==NVT equillibration== and ==NPT production MD== condition.
    - `1*bar`: bar is a standard measure of pressure, and 1 bar is approximately equal to atmospheric pressure at sea level.

!!! note What is NVT and NPT?
    <font title='ChatGPT o1' color=gray>In the NVT (constant Number of particles, Volume, and Temperature) ensemble, the system is thermally equilibrated at a fixed volume to achieve a stable temperature distribution. This step ensures that any initial structural distortions and non-equilibrium distributions of kinetic energy dissipate, providing a well-relaxed starting point. Following NVT equilibration, the system is often subjected to an NPT (constant Number of particles, Pressure, and Temperature) ensemble, where both temperature and pressure are maintained constant. This allows the simulation box volume to fluctuate to the pressure target, enabling the system’s density and structure to equilibrate under more experimentally relevant conditions. The transition from NVT to NPT thus facilitates a smooth pathway from initial equilibration to realistic production conditions, offering a balanced and physically representative environment for subsequent analyses of structural, thermodynamic, and dynamic properties.</font>

| Feature              | NVT Equilibration          | NPT Production MD          |
|----------------------|----------------------------|----------------------------|
| Ensemble             | Canonical (NVT)            | Isothermal–Isobaric (NPT)  |
| Variables Held Fixed | Number of particles (N), Volume (V), Temperature (T) | Number of particles (N), Pressure (P), Temperature (T) |
| Volume Adjustment    | Fixed volume               | Volume fluctuates to maintain target pressure |
| Pressure Control     | Not controlled, can fluctuate | Actively controlled via a barostat |
| Typical Use          | Initial temperature equilibration after energy minimization | Production runs to simulate conditions resembling experimental environments |
| Realism              | Less physically representative of ambient conditions (volume fixed) | More realistic: system adapts to pressure, resulting in stable density |
| Common Duration      | Shorter (tens to hundreds of picoseconds) | Longer (nanoseconds to microseconds) for data collection |
| Outcome              | Thermally equilibrated structure at given T | Equilibrium structure and dynamics at given P and T, suitable for analysis |


## Protein Relaxation Test

```python
from sys import stdout
from openmm.app import *
from openmm import *
from openmm.unit import *

# Input files
pdb_filename = 'best_fix.pdb'  # Your starting protein structure (from cryo-EM)
forcefield_files = ['amber14-all.xml', 'amber14/tip3pfb.xml']  # Force fields
ionic_strength = 0.15*molar

# Load the PDB
pdb = PDBFile(pdb_filename)

# Create a forcefield object
forcefield = ForceField(*forcefield_files)

# Create a model of the system with solvent
# Add a water box around the protein (10 Å padding)
modeller = Modeller(pdb.topology, pdb.positions)
#modeller.addSolvent(forcefield, model='tip3p', boxSize=Vec3(10,10,20)*nanometer, ionicStrength=ionic_strength)
modeller.addSolvent(forcefield, padding=1.0*nanometer, ionicStrength=ionic_strength)

# Create the system
system = forcefield.createSystem(
    modeller.topology,
    nonbondedMethod=PME,
    nonbondedCutoff=1.0*nanometer,
    constraints=HBonds,
    hydrogenMass=4*amu
)

# Add a thermostat and barostat for later (NPT)
temperature = 300*kelvin
pressure = 1*bar
friction = 1/picosecond
timestep = 0.002*picoseconds

system.addForce(MonteCarloBarostat(pressure, temperature))

# Create integrator (for equilibration and production)
integrator = LangevinIntegrator(temperature, friction, timestep)

# Create simulation object
platform = Platform.getPlatformByName('CUDA')  # or 'CUDA'/'OpenCL' if available
simulation = Simulation(modeller.topology, system, integrator, platform)
simulation.context.setPositions(modeller.positions)

# Minimization
print("Minimizing...")
simulation.minimizeEnergy(maxIterations=1000)

# save the last structure from simulation
state = simulation.context.getState(getPositions=True)
positions = state.getPositions()
# Write out the final structure to a PDB file
with open('relax_test_mini.pdb', 'w') as f:
    PDBFile.writeFile(simulation.topology, positions, f)
    
# NVT Equilibration: Remove barostat and fix volume for initial temp equilibration
# (Optional step: you can also start directly with NPT if you prefer)
forces = { force.__class__.__name__: force for force in system.getForces() }
system.removeForce(list(forces.keys()).index('MonteCarloBarostat'))
simulation.context.reinitialize(preserveState=True)

simulation.context.setVelocitiesToTemperature(temperature)
print("Equilibrating under NVT conditions...")
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True))
simulation.reporters.append(DCDReporter('relax.dcd', 1000))  # Save a frame every 1000 steps

print('start simulation')
simulation.step(50000)  # ~100 ps of NVT equilibration (adjust as needed)

# save the last structure from simulation
state = simulation.context.getState(getPositions=True)
positions = state.getPositions()
# Write out the final structure to a PDB file
with open('relax_test_NVT.pdb', 'w') as f:
    PDBFile.writeFile(simulation.topology, positions, f)

# Re-introduce NPT conditions (barostat)
system.addForce(MonteCarloBarostat(pressure, temperature))
simulation.context.reinitialize(preserveState=True)

print("Equilibrating under NPT conditions...")
# Remove old reporters and add a new one
simulation.step(50000)  # Another ~100 ps (adjust as needed)

# save the last structure from simulation
state = simulation.context.getState(getPositions=True)
positions = state.getPositions()
# Write out the final structure to a PDB file
with open('relax_test_NPT.pdb', 'w') as f:
    PDBFile.writeFile(simulation.topology, positions, f)

# Now we have an equilibrated system at NPT.
# This is where you might start your production run.

production_steps = 250000  # ~500 ps of production (adjust as needed)
#simulation.reporters.append(PDBReporter('output_production.pdb', 5000)) # Save frames every 10 ps
simulation.reporters.append(StateDataReporter('production_log.csv', 1000, step=True, time=True, potentialEnergy=True,
                                               kineticEnergy=True, totalEnergy=True, temperature=True,
                                               volume=True, density=True))

print("Running Production MD...")
simulation.step(production_steps)
print("Done!")

state = simulation.context.getState(getPositions=True)
positions = state.getPositions()
# Write out the final structure to a PDB file
with open('relax_test_final.pdb', 'w') as f:
    PDBFile.writeFile(simulation.topology, positions, f)
```

In the final simulation, your protein may on the "edge" of the box. So, we need to adjust the relative position of the protein

|Before Recenter|After Recenter|
|:-:|:-:|
|![](https://imgur.com/1obBq2B.png)|![](https://imgur.com/kyKbB8I.png)|

## Recenter the Protein

```python
import mdtraj as md

# Load the trajectory and topology
traj = md.load('relax.dcd', top='relax_test_mini.cif')
traj = traj.image_molecules()

# Re-center coordinates so the protein is centered in the box
centered_traj = traj.center_coordinates()
# Save the re-centered trajectory as a multi-model PDB (one MODEL per frame)
#centered_traj.save_pdb('centered_system.pdb')
centered_traj.save_dcd('centered_system.dcd')
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
