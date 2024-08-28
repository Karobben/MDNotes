---
toc: true
url: DLVOtheory
covercopy: © Karobben
priority: 10000
date: 2024-08-16 16:27:54
title: "DLVO theory: Atom Interaction"
ytitle: "DLVO theory: Atom Interaction"
description: "DLVO theory describes the forces between charged surfaces interacting through a liquid medium. The theory combines two main types of forces"
excerpt: "DLVO theory is named after Derjaguin, Landau, Verwey, and Overbeek, who developed it in the 1940s. It describes the forces between charged surfaces interacting through a liquid medium. The theory combines two main types of forces"
tags: [Physics]
category: [Notes, other]
cover: "https://imgur.com/GMxKF8T.png"
thumbnail: "https://imgur.com/GMxKF8T.png"
---

## DLVO Theory

|![DLVO theory](https://ars.els-cdn.com/content/image/3-s2.0-B0080431526016223-gr3.gif)|
|:-:|
|[© J.H. Adair; 2001](https://www.sciencedirect.com/science/article/abs/pii/B0080431526016223)|

DLVO theory is named after Derjaguin, Landau, Verwey, and Overbeek, who developed it in the 1940s. It describes the forces between charged surfaces interacting through a liquid medium. The theory combines two main types of forces:

1. **Van der Waals forces:** These are attractive forces that arise from induced electrical interactions between molecules or atoms.
2. **Electrostatic double-layer forces:** These are repulsive forces that occur due to the overlap of electrical double layers surrounding charged particles.

The balance between these forces determines whether particles will aggregate (if the attractive forces dominate) or remain stable in suspension (if the repulsive forces dominate). This theory is widely used in colloid chemistry, environmental science, and materials science to understand and predict the stability of colloidal dispersions.

The DLVO theory describes the interaction energy $ U_{total} $ between two colloidal particles as the sum of the Van der Waals attraction $ U_{VdW} $ and the electrostatic repulsion $ U_{elec} $. The general form of the DLVO potential is given by:

$$ U_{total}(h) = U_{VdW}(h) + U_{elec}(h) $$

where $ h $ is the distance between the surfaces of the particles.

### Van der Waals Attraction ($ U_{VdW} $)

The Van der Waals attraction energy between two spherical particles of radius $ R $ at a separation distance $ h $ is given by:

$$ U_{VdW}(h) = - \frac{A}{6} \left( \frac{2R^2}{h(2R + h)} + \frac{2R^2}{(2R + h)^2} + \ln \left( \frac{h}{2R + h} \right) \right) $$

- $ A $: Hamaker constant (typically around $ 10^{-20} $ J for biological systems)
- $ R $: Radius of the amino acid (approx. $ 0.5 $ nm or $ 0.5 \times 10^{-9} $ m)
- $ h $: Separation distance between the particles


!!! note What if the atoms are different?
    $R^2= R_1 * R_2$
    $2R = R_1 + R_2$ 

### Electrostatic Repulsion ($ U_{elec} $)

The electrostatic repulsion energy between two spherical particles with surface potential $ \psi_0 $ and radius $ R $ in a medium with Debye length $ \kappa^{-1} $ (which is related to the ionic strength of the medium) is given by:

$$ U_{elec}(h) = 2 \pi \epsilon R \psi_0^2 \ln \left( 1 + \exp(-\kappa h) \right) $$


- $ \epsilon $: Permittivity of the medium (water, typically $ 80 \times 8.854 \times 10^{-12} $ F/m)
- $ \psi_0 $: Surface potential (approx. $ 25 $ mV or $ 25 \times 10^{-3} $ V)
- $ \kappa $: Inverse Debye length (for a Debye length of $ 1 $ nm, $ \kappa \approx 10^9 $ m$^{-1}$)
- $ h $: Separation distance between the particles


### Total Interaction Energy

Combining these two expressions, the total interaction energy is:

$$ U_{total}(h) = - \frac{A}{6} \left( \frac{2R^2}{h(2R + h)} + \frac{2R^2}{(2R + h)^2} + \ln \left( \frac{h}{2R + h} \right) \right) + 2 \pi \epsilon R \psi_0^2 \ln \left( 1 + \exp(-\kappa h) \right) $$

This equation allows us to predict whether the colloidal particles will repel each other and remain stable in suspension or attract each other and aggregate, depending on the balance of the attractive and repulsive forces.

### Separation Distance (h) 

- If the radii $ R_1 $ and $ R_2 $ of two spherical particles are known, and the center-to-center distance between them is $ D $, the separation distance $ h $ is calculated as:
 $$
 h = D - (R_1 + R_2)
 $$
- For identical particles with the same radius $ R $, it simplifies to:
 $$
 h = D - 2R
 $$

In the code snippet provided, the parameters can be categorized into **constant parameters** (those that remain the same across different residues) and **variable parameters** (those that may change depending on the specific residues or the system under consideration).

## Parameters

### Constant Parameters:

1. **$ A $** (Hamaker constant): 
   - **Value:** $ 1 \times 10^{-20} $ J
   - **Description:** This is a material-specific constant that depends on the nature of the interacting particles and the medium. For biological molecules in water, it's often taken as a constant.

2. **$ \epsilon $** (Permittivity of the medium):
   - **Value:** $ 80 \times 8.854 \times 10^{-12} $ F/m (Permittivity of water)
   - **Description:** The permittivity of the medium (usually water in biological contexts) is a constant based on the dielectric properties of the solvent.

3. **$ \kappa $** (Inverse Debye length):
   - **Value:** $ 1 \times 10^9 $ m$^{-1}$
   - **Description:** The inverse Debye length is related to the ionic strength of the medium and is often considered constant under specific conditions, such as physiological ionic strength.

### Variable Parameters:

1. **$ R $** (Radius of the amino acid):
   - **Value:** $ 0.5 \times 10^{-9} $ m (0.5 nm)
   - **Description:** The radius could vary slightly between different amino acids, especially when considering side chains. The value used here is an approximation and might need adjustment for specific residues.

2. **$ \psi_0 $** (Surface potential):
   - **Value:** $ 25 \times 10^{-3} $ V (25 mV)
   - **Description:** The surface potential can vary depending on the charge state of the amino acid side chains. For example, charged residues like lysine or aspartic acid will have different surface potentials compared to neutral residues like alanine.

3. **$ h $** (Separation distance):
   - **Value:** Range from $ 0.1 \times 10^{-9} $ m to $ 10 \times 10^{-9} $ m
   - **Description:** The separation distance between two residues or atoms is the primary variable in these calculations, often determined by the 3D structure of the protein or molecular complex being studied.

## Who to Know the R and h?

To calculate the radius of amino acids such as valine (V) and phenylalanine (F), you're generally referring to an approximation of the **van der Waals (VDW) radius** or the **effective radius** of the entire amino acid side chain. This radius can be used in models like DLVO theory to represent the size of the interacting particle.

### Methods to Determine the Radius:

1. **Van der Waals Radius of Atoms:**
   - The van der Waals radius is an inherent property of atoms and can be summed up to approximate the radius of a molecule or side chain. For example, the VDW radius for carbon is about 1.7 Å, and for hydrogen, it's about 1.2 Å.

2. **Effective Radius from Crystal Structures:**
   - If you have a crystal structure or molecular model, you can measure the effective radius of the side chain by considering the spatial extent of the side chain atoms. This is often done using software tools that can calculate the solvent-accessible surface area (SASA) or by directly measuring distances in a molecular viewer.

3. **Using Approximate Values from Literature:**
   - For many applications, approximate radii for amino acids are available in the literature based on their typical side chain sizes.

### Approximate Radii for Valine (V) and Phenylalanine (F):

- **Valine (V):** 
  - Valine has a branched, non-polar side chain. Its effective radius is often approximated as **~3.0 Å (0.3 nm)**.
  
- **Phenylalanine (F):**
  - Phenylalanine has a larger, aromatic side chain. Its effective radius is typically around **~3.5-4.0 Å (0.35-0.4 nm)**.

These values are not exact but are generally used in theoretical calculations.

### Calculation Example:

If you need to calculate the interaction between valine (V) and phenylalanine (F), you could use these approximate radii:

- **Valine (V):** $ R_V \approx 0.3 $ nm
- **Phenylalanine (F):** $ R_F \approx 0.35 $ nm

The separation distance $ h $ would then be calculated based on the center-to-center distance $ D $ between the residues:

$$
h = D - (R_V + R_F)
$$

If $ D $ is known (from a crystal structure or a model), this formula gives the separation distance $ h $ between the residues' surfaces.

### Tools for More Accurate Measurements:

- **Molecular Visualization Software (e.g., PyMOL, Chimera):** You can load a protein structure and measure the distance between specific atoms or calculate the van der Waals surface.
- **Computational Tools:** Software packages like CHARMM, AMBER, or GROMACS can provide detailed calculations based on molecular dynamics or energy minimization, giving more precise values for radii in specific contexts.


## Python Script

```python
import numpy as np
import matplotlib.pyplot as plt

def van_der_waals(h, A, R):
    """Calculate Van der Waals attraction energy."""
    term1 = (2 * R**2) / (h * (2 * R + h))
    term2 = (2 * R**2) / (2 * R + h)**2
    term3 = np.log(h / (2 * R + h))
    return - (A / 6) * (term1 + term2 + term3)

def electrostatic_repulsion(h, epsilon, R, psi_0, kappa):
    """Calculate electrostatic repulsion energy."""
    return 2 * np.pi * epsilon * R * psi_0**2 * np.log(1 + np.exp(-kappa * h))

def dlvo_total(h, A, R, epsilon, psi_0, kappa):
    """Calculate total DLVO interaction energy."""
    return van_der_waals(h, A, R) + electrostatic_repulsion(h, epsilon, R, psi_0, kappa)

# Parameters
A = 1e-20  # Hamaker constant in J
R = 1e-7   # Radius of particles in m
epsilon = 80 * 8.854e-12  # Permittivity of water in F/m
psi_0 = 25e-3  # Surface potential in V
kappa = 1e8    # Inverse Debye length in 1/m
h = np.linspace(5e-10, 1e-7, 400)  # Separation distance in m

# Calculate DLVO potential
U_vdw = van_der_waals(h, A, R)
U_elec = electrostatic_repulsion(h, epsilon, R, psi_0, kappa)
U_total = dlvo_total(h, A, R, epsilon, psi_0, kappa)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(h * 1e9, U_vdw, label='Van der Waals Attraction', linestyle='-.')
plt.plot(h * 1e9, U_elec, label='Electrostatic Repulsion', linestyle='-.')
plt.plot(h * 1e9, U_total, label='Total DLVO Potential')
plt.xlabel('Separation Distance (nm)')
plt.ylabel('Interaction Energy (J)')
plt.title('DLVO Theory Interaction Energy')
plt.legend()
plt.grid(True)
plt.show()
```

![](https://imgur.com/GMxKF8T.png)

## Simplified System

### **Empirical Formula in the Context of Your Problem**

A general empirical formula for the interaction energy $ \Delta G_{i,j} $ between two residues $ H_i $ and $ A_j $ might look like this:

$$
\Delta G_{i,j} = V_{\text{LJ}}(r_{ij}) + V_{\text{Coulomb}}(r_{ij})
$$

Where:
- $ r_{ij} $ is the distance between residue $ H_i $ and residue $ A_j $.
- $ V_{\text{LJ}}(r_{ij}) $ represents the van der Waals interaction.
- $ V_{\text{Coulomb}}(r_{ij}) $ represents the electrostatic interaction.

### **Lennard-Jones Potential (van der Waals Interactions)**

The Lennard-Jones potential is a commonly used empirical formula to describe the van der Waals forces between two non-bonded atoms or molecules. It has the form:

$$
V_{\text{LJ}}( r ) = 4\epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right]
$$

- $ V_{\text{LJ}}( r ) $ is the potential energy as a function of distance $ r $ between two particles.
- $ \epsilon $ is the depth of the potential well, representing the strength of the interaction.
- $ \sigma $ is the distance at which the potential energy is zero (often related to the size of the atoms/molecules).
- $ r $ is the distance between the two particles.

The term $ \left(\frac{\sigma}{r}\right)^{12} $ represents the repulsive interaction at short distances (due to Pauli exclusion principle), and the term $ \left(\frac{\sigma}{r}\right)^{6} $ represents the attractive van der Waals forces at longer distances.

### **Coulomb's Law (Electrostatic Interactions)**

Coulomb's law describes the electrostatic interaction between two charged particles:

$$
V_{\text{Coulomb}}( r ) = \frac{k_e \cdot q_1 \cdot q_2}{r}
$$

- $ V_{\text{Coulomb}}( r ) $ is the potential energy between two charges.
- $ k_e $ is Coulomb's constant ($ 8.9875 \times 10^9 \, \text{N} \cdot \text{m}^2/\text{C}^2 $ in vacuum).
- $ q_1 $ and $ q_2 $ are the charges of the two interacting particles.
- $ r $ is the distance between the two charges.

### Python Script

```python
from prody import *
import numpy as np

# Load the protein structure
structure = parsePDB('your_structure.pdb')

# Select the side chains of the residues of interest
residue1_sidechain = structure.select('resid 10 and sidechain')
residue2_sidechain = structure.select('resid 20 and sidechain')

# Function to calculate interaction energy considering only side chains
def calculate_interaction_energy(residue1, residue2, cutoff=5.0):
    """
    Calculate the van der Waals and electrostatic interaction energy 
    between two residues' side chains using a simple empirical formula.
    
    :param residue1: ProDy atom group for the first residue side chain
    :param residue2: ProDy atom group for the second residue side chain
    :param cutoff: Distance cutoff for interaction (in Å)
    :return: Tuple of (vdW energy, electrostatic energy)
    """
    # van der Waals parameters (simplified example)
    epsilon = 0.1  # Depth of the potential well (kcal/mol)
    sigma = 3.5  # Distance at which the potential is zero (Å)
    
    # Coulomb constant (for electrostatic energy calculation)
    k_e = 8.9875517873681764e9  # N m² C⁻² (can be adjusted for unit compatibility)
    
    # Simplified charges for electrostatic calculation
    charge1 = np.sum([atom.getCharge() for atom in residue1])
    charge2 = np.sum([atom.getCharge() for atom in residue2])
    
    vdW_energy = 0.0
    electrostatic_energy = 0.0

    # Calculate pairwise interactions considering only side chains
    for atom1 in residue1:
        for atom2 in residue2:
            distance = np.linalg.norm(atom1.getCoords() - atom2.getCoords())
            if distance < cutoff:
                # van der Waals energy (Lennard-Jones potential)
                vdW_energy += 4 * epsilon * ((sigma / distance)**12 - (sigma / distance)**6)
                # Electrostatic energy (Coulomb's law)
                electrostatic_energy += k_e * (charge1 * charge2) / distance
    return vdW_energy, electrostatic_energy

# Calculate interaction energy
vdW_energy, electrostatic_energy = calculate_interaction_energy(residue1_sidechain, residue2_sidechain)
print(f"van der Waals Energy: {vdW_energy:.2f} kcal/mol")
print(f"Electrostatic Energy: {electrostatic_energy:.2f} kcal/mol")
```

<details><summary>Codes for the plot</summary>

```python
h = np.linspace(3.3, 10, 400)  # Separation distance in m
# Plot results
plt.figure(figsize=(10, 6))
plt.plot(h, [0 for i in h], linestyle = '-.')
plt.plot(h, [calculate_interaction_energy(i)  for i in h], color = 'salmon')
plt.xlabel('Separation Distance (Åm)')
plt.ylabel('Interaction Energy')
plt.title('Lennard-Jones Potential')
plt.legend()
plt.grid(True)
plt.show()
```
</details>

|![](https://imgur.com/wmcPi2K.png)|
|:-:|
|In this plot, it shows the change of the Lennard-Jones Potential with the change of the distance when the $\sigma = 3.5$ |



<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>
