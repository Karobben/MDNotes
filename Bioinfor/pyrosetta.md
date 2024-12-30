---
toc: true
url: pyrosetta
covercopy: © Karobben
priority: 10000
date: 2024-12-20 18:06:54
title: "pyrosetta" 
ytitle: "pyrosetta"
description: "pyrosetta"
excerpt: "pyrosetta"
tags: []
category: []
cover: "https://imgur.com/l41Fw8X.png"
thumbnail: "https://imgur.com/l41Fw8X.png"
---


## Loop Regenerate (Codes From ChatGPT)

```python
from pyrosetta import init, Pose, get_fa_scorefxn
from pyrosetta.rosetta.protocols.loops import Loops, Loop
from pyrosetta.rosetta.protocols.loops.loop_mover.perturb import LoopMover_Perturb_KIC
from pyrosetta.rosetta.protocols.loops.loop_mover.refine import LoopMover_Refine_KIC, LoopMover_Refine_CCD

#from pyrosetta.rosetta.core.import_pose import pose_from_pdbstring as pose_from_pdb
from pyrosetta import  pose_from_pdb

# 1. Initialize PyRosetta
init()

# 2. Load Your Protein Pose
pose = pose_from_pdb( "test.pdb")

n_chains = pose.num_chains()
for chain_index in range(1, n_chains+1):
    start_res = pose.chain_begin(chain_index)
    end_res = pose.chain_end(chain_index)
    print(f"Chain {chain_index}: residues {start_res} to {end_res}")

loops_by_chain = {}

# Iterate over chains
n_chains = pose.num_chains()
for chain_index in range(1, n_chains+1):
    start_res = pose.chain_begin(chain_index)
    end_res = pose.chain_end(chain_index)

    # Extract secondary structure substring for this chain
    chain_secstruct = secstruct[start_res-1:end_res]

    loop_regions = []
    current_loop_start = None

    # Identify loop regions as stretches of 'L'
    for i, s in enumerate(chain_secstruct, start=start_res):
        if s == 'L':
            if current_loop_start is None:
                current_loop_start = i
        else:
            if current_loop_start is not None:
                loop_regions.append((current_loop_start, i-1))
                current_loop_start = None

    # Check if a loop extends to the end of the chain
    if current_loop_start is not None:
        loop_regions.append((current_loop_start, end_res))

    # Extract sequences for each loop region
    # Store them in a dictionary keyed by chain index
    chain_loops = []
    for (loop_start, loop_end) in loop_regions:
        # Extract the sequence of the loop
        loop_seq = "".join([pose.residue(r).name1() for r in range(loop_start, loop_end+1)])
        chain_loops.append({
            "start": loop_start,
            "end": loop_end,
            "sequence": loop_seq
        })

    loops_by_chain[chain_index] = chain_loops

n_chains = pose.num_chains()
for chain_index in range(1, n_chains+1):
    start_res = pose.chain_begin(chain_index)
    end_res = pose.chain_end(chain_index)
    print(f"Chain {chain_index}: residues {start_res} to {end_res}")

# 3. Define the Loop(s) You Want to Remodel
# Suppose you want to remodel the loop from residues 45 to 55.
# Choose a cut point (ideally inside the loop), typically near the middle.
loop_start = 593
loop_end = 608
cutpoint = 601

loops = Loops()
loops.add_loop( Loop(loop_start, loop_end, cutpoint) )

# 4. Set Up a Scorefunction
scorefxn = get_fa_scorefxn()

# 5. Set Up the Loop Remodeling Protocol
# You have multiple options: 
# Example: Use KIC Perturb and then Refine
perturb_mover = LoopMover_Perturb_KIC(loops)
perturb_mover.set_scorefxn(scorefxn)

refine_mover = LoopMover_Refine_KIC(loops)
refine_mover.set_scorefxn(scorefxn)

# Alternatively, you might use CCD refinement:
# refine_mover = LoopMover_Refine_CCD(loops)
# refine_mover.set_scorefxn(scorefxn)

# 6. Optionally: Set Up Monte Carlo or Repeats
# Often you do multiple trials and pick the best model.

# 7. Apply the Movers
# First do perturbation
perturb_mover.apply(pose)

# Then refine
refine_mover.apply(pose)

# After this, you should have a remodeled loop region.
# You can save the resulting structure to a PDB file:
pose.dump_pdb("remodeled_loop.pdb")
```

|||
| :--: | :------- |
| ![Raw loop](https://imgur.com/eJG1S0x.png) | Raw loop |
| ![Predicted loop](https://imgur.com/Lu1BdXo.png)| Predicted loop by ussing ImmuneBuilder. The Predicted results has some trouble in the CDRH3 region. And if we place it in the corrected position and it has crush. |
|![Reconstructed loop](https://imgur.com/l41Fw8X.png)| Rosetta reconstructed loop by using the code above. Rosetta takes lots of time to reconstruct the loop and the result is terrible. The loop inseted into a very wired and unlikly position|

## Loop Regenerate (Codes From Tutorial)

![](https://imgur.com/1Op8NKe.png)

In the Tutorial 9.01, it use 2 structure: 1) the complete structure and 2) the structure has gap. The missing parts is range from 29~31. It not only deleted 5 residues, but also split it into 2 chains.

Because it was in a separate chain, the index 28 and 29 is the C terminal and N terminal in the chain with gap. The selected residues is 28 and 32 when they are in the original structure

### How it works in antibody CDRH3

```python
# Notebook setup
import pyrosettacolabsetup; pyrosettacolabsetup.install_pyrosetta()
import pyrosetta; pyrosetta.init()

# py3Dmol setup (if there's an error, make sure you have 'py3Dmol' and 'ipywidgets' pip installed)
import glob
import logging
logging.basicConfig(level=logging.INFO)
import os
import pyrosetta.distributed
import pyrosetta.distributed.io as io
import pyrosetta.distributed.viewer as viewer
from pyrosetta import pose_from_pdb

input_pose = pose_from_pdb('/mnt/Data/PopOS/Data_Ana/Wu/PigAntiBodies/AB_regine/data/14-1_ImmuneCorrect_partial.pdb')
input_pose_no_loop = pose_from_pdb('/mnt/Data/PopOS/Data_Ana/Wu/PigAntiBodies/AB_regine/data/14-1_ImmuneCorrect_partial_Noloop.pdb')


helix_selector = pyrosetta.rosetta.core.select.residue_selector.SecondaryStructureSelector("H")
loop_selector = pyrosetta.rosetta.core.select.residue_selector.SecondaryStructureSelector("L")

modules = [
    viewer.setBackgroundColor(color="black"),
    viewer.setStyle(residue_selector=helix_selector, cartoon_color="blue", label=False, radius=0),
    viewer.setStyle(residue_selector=loop_selector, cartoon_color="yellow", label=False, radius=0),
    viewer.setZoomTo(residue_selector=loop_selector)
]

#view = viewer.init(input_pose, window_size=(800, 600), modules=modules).show()
#view = viewer.init(input_pose_no_loop, window_size=(800, 600), modules=modules).show()

def Chain_num(pose):
    n_chains = pose.num_chains()
    for chain_index in range(1, n_chains+1):
        start_res = pose.chain_begin(chain_index)
        end_res = pose.chain_end(chain_index)
        print(f"Chain {chain_index}: residues {start_res} to {end_res}")

Chain_num(input_pose)
Chain_num(input_pose_no_loop)

Start = input_pose_no_loop.chain_end(1)
Len = input_pose.chain_end(1) - input_pose_no_loop.chain_end(2)
End = Start + Len
Miss_Seq = "".join([input_pose.residue(i).name1() for i in range(Start, End)])

##The c terminus of one helix
print(input_pose_no_loop.residue(Start).name())
#The N terminus of the other helix
print(input_pose_no_loop.residue(Start+1).name())

def mutate_position(pose,position,mutate):
    '''A simple function to mutate an amino acid given a pose number'''
    mr = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
    mr.set_target(position)
    mr.set_res_name(mutate)
    mr.apply(pose)


##Mutate both 28 and 29 to ALA
'''
In tutorial, they mutated the residues into ALA. It says is make a pose that can be applied by GenKic. I am not sure it is because of the GenKic algorithem or GenKic function. I possible could be that GenKic function doesn't accept the resideus from either side do the termianl. So, we need to remove them and add them back. Because my goal is get a better conformation of the loop, I just relpace it with itself to do the lateral test.
Also, I tried to use ALA at the begining, too. The folding resutls are not promissing.
'''
Resi1 = input_pose_no_loop.residue(Start).name3()
Resi2 = input_pose_no_loop.residue(Start+1).name3()

mutate_position(input_pose_no_loop,Start,Resi1)
mutate_position(input_pose_no_loop,Start+1,Resi2)
assert(input_pose_no_loop.residue(Start).name() == Resi1)
assert(input_pose_no_loop.residue(Start+1).name() == Resi2)

from pyrosetta import Pose

def slice_pose(p,start,end):
    '''
    Take a pose object and return from start, end
    '''
    sliced = Pose()
    if end > p.size() or start > p.size():
        return "end/start slice is longer than total lenght of pose {} {}".format(start,end)
    for i in range(start,end+1):
        sliced.append_residue_by_bond(p.residue(i))
    return sliced

##Pose object 1 - helix_AB all the way up to residue 28
helix_ab_pose = slice_pose(input_pose_no_loop,1,Start)
##Pose object 2 - helix C and the reaminder of the pose
#helix_c_pose = slice_pose(input_pose_no_loop,Start+1,input_pose_no_loop.size())
helix_c_pose = slice_pose(input_pose_no_loop,Start+1,input_pose_no_loop.chain_end(2))

# We're just going to quicky add in pdb info so that our viewing commands work
add_pdb_info_mover = pyrosetta.rosetta.protocols.simple_moves.AddPDBInfoMover()
add_pdb_info_mover.apply(helix_ab_pose)
add_pdb_info_mover.apply(helix_c_pose)
# Here's the second part
#view = viewer.init(helix_c_pose, window_size=(800, 600), modules=modules).show()
# Here's the first object
#view = viewer.init(helix_ab_pose, window_size=(800, 600), modules=modules).show()

# Here's the second object
#view = viewer.init(helix_c_pose, window_size=(800, 600), modules=modules).show()
def crudely_connect_w_loop(n_term_pose,c_term_pose,connect_with):
    """
    The function will take two poses and join them with a loop

    Keep in mind this is just joined as far as the pose is concerned. The bond angles and lenghts will be sub-optimal
    """
    one_to_three = {
    'A': 'ALA',
    'C': 'CYS',
    'D': 'ASP',
    'E': 'GLU',
    'F': 'PHE',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'K': 'LYS',
    'L': 'LEU',
    'M': 'MET',
    'N': 'ASN',
    'P': 'PRO',
    'Q': 'GLN',
    'R': 'ARG',
    'S': 'SER',
    'T': 'THR',
    'V': 'VAL',
    'Y': 'TYR',
    'W': 'TRP'}

    pose_a = Pose()
    pose_a.assign(n_term_pose)

    pose_b = Pose()
    pose_b.assign(c_term_pose)

    # Setup CHEMICAL MANAGER TO MAKE NEW RESIDUES
    chm = pyrosetta.rosetta.core.chemical.ChemicalManager.get_instance()
    rts = chm.residue_type_set('fa_standard')
    get_residue_object = lambda x: pyrosetta.rosetta.core.conformation.ResidueFactory.create_residue(
        rts.name_map(x))

    # Will keep track of indexing of rebuilt loop
    rebuilt_loop = []

    '''Iterate through string turning each letter into a residue object and then
    appending it to the N term pose'''
    for one_letter in connect_with:
        resi = get_residue_object(one_to_three[one_letter])
        pose_a.append_residue_by_bond(resi, True)
        pose_a.set_omega(pose_a.total_residue(), 180.)
        rebuilt_loop.append(pose_a.total_residue())

    ##ADD the C term pose to the end of the loop we just appended
    for residue_index in range(1, pose_b.total_residue()+1):
        pose_a.append_residue_by_bond(
            pose_b.residue(residue_index))

    print("Joined NTerm and CTerm pose with loop {} at residues {}".format(connect_with,rebuilt_loop))
    return pose_a

#Returns a pose that is connected, but sub-optimal geometry
gk_input_pose = crudely_connect_w_loop(helix_ab_pose,helix_c_pose,Miss_Seq)
#gk_input_pose = crudely_connect_w_loop(helix_ab_pose,helix_c_pose,Miss_Seq)

print(Miss_Seq)
for chain in range(3,input_pose_no_loop.num_chains()+1):
    print(chain)
    gk_input_pose.append_pose_by_jump(input_pose_no_loop.split_by_chain(chain), gk_input_pose.total_residue())

Chain_num(helix_ab_pose)
Chain_num(gk_input_pose)
Chain_num(input_pose)



from additional_scripts.GenKic import GenKic

##All that GenKic needs is the loop residue list
loop_residues = [i for i in range(Start,End+2)]
gk_object = GenKic(loop_residues)

##Let's set the closure attempt to 500000
gk_object.set_closure_attempts(500000)
gk_object.set_min_solutions(10)


from pyrosetta import ScoreFunction

def get_bb_only_sfxn():
    scorefxn = ScoreFunction()
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.fa_atr, 1)    # full-atom attractive score
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.fa_rep, 0.55)    # full-atom repulsive score
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.hbond_sr_bb, 1)    # short-range hbonding
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.hbond_lr_bb, 1)    # long-range hbonding
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.rama_prepro, 0.45)    # ramachandran score
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.omega, 0.4)    # omega torsion score
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.p_aa_pp, 0.625)
    return scorefxn

##Grab BB Only SFXN
bb_only_sfxn = get_bb_only_sfxn()

##Pass it to GK
gk_object.set_scorefxn(bb_only_sfxn)

gk_object.set_selector_type('lowest_energy_selector')
#First lets set alll mainchain omega values to 180 degrees in our loop. We don't want to include residue after the last anchor residue as that could potentially not exist.
for res_num in loop_residues[:-1]:
    gk_object.set_dihedral(res_num, res_num + 1, "C", "N", 180.1)

###Or there is a convienience function within the class that does the same thing
gk_object.set_omega_angles()

for res_num in loop_residues:
    gk_object.randomize_backbone_by_rama_prepro(res_num)
##This will grab the GK instance and apply everything we have set to our pose
gk_object.get_instance().apply(gk_input_pose)

##You can see, we perturbed the loop, but we did not tell GK to close the bond
#view = viewer.init(gk_input_pose, window_size=(800, 600), modules=modules).show()

gk_object.close_normal_bond(End,End+1) #or gk_object.close_normal_bond(loop_residues[-2],loop_residues[-1])
gk_object.get_instance().apply(gk_input_pose)

#view = viewer.init(gk_input_pose, window_size=(800, 600), modules=modules).show()

##The first residue in our loop definition will be confiend to alpha-helical rama space
gk_object.set_filter_backbone_bin(loop_residues[0],'A',bin='ABBA')
##The last residue in our loop definition will be confiend to alpha-helical rama space
gk_object.set_filter_backbone_bin(loop_residues[-1],'A',bin='ABBA')

gk_object.set_filter_loop_bump_check()


for r in gk_object.pivot_residues:
    gk_object.set_filter_rama_prepro(r,cutoff=0.5)

##Grab GK instance
gk_instance = gk_object.get_instance()
##apply it to the pose
gk_instance.apply(gk_input_pose)
##The Input pose with no loop, the reference pose we are trying to recreate and the GK pose
poses = [input_pose_no_loop, input_pose, gk_input_pose]
# view = viewer.init(poses) + viewer.setStyle()
# view()

#gk_input_pose.dump_pdb("final_pose.pdb")

from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.kinematics import MoveMap

from pyrosetta.rosetta.protocols.relax import FastRelax



gk_input_pose.dump_pdb("loop.pdb")

# loop alone
relax = FastRelax()
relax.set_scorefxn(bb_only_sfxn)
movemap = MoveMap()
#for res in loop_residues:
#  movemap.set_bb(res, True)
relax.set_movemap(movemap)
relax.apply(gk_input_pose)


gk_input_pose.dump_pdb("loop_relaxe.pdb")

# relax: Full
relax = FastRelax()
relax.set_scorefxn(bb_only_sfxn)
relax.apply(gk_input_pose)


gk_input_pose.dump_pdb("loop_full_relaxe.pdb")
```

In this script, I am not only test the loop reconstruction, but also add relaxation steps. Here is the results from different methods. It seems like no matter how you try, it is hard to reconstruct this loop in Rosetta.

|||
|:-:|:-:|
|![Reconstructed Loop](https://imgur.com/iyjWPH6.png)|![Loop-reconstruction and Relaxation](https://imgur.com/ZX450qV.png)|
|![Relaxation only](https://imgur.com/wTwR6P5.png)|![Realxation and then loop rexonstruction](https://imgur.com/BRtUTD1.png)|


## How to check the Chain and the number of residues

```python
from pyrosetta import init, pose_from_pdb

# 1. Initialize PyRosetta
init()
# 2. Load Your Protein Pose
pose = pose_from_pdb( "data/14-1_ImmuneCorrect.pdb")

# 3. Count and print the result

n_chains = pose.num_chains()
for chain_index in range(1, n_chains+1):
    start_res = pose.chain_begin(chain_index)
    end_res = pose.chain_end(chain_index)
    print(f"Chain {chain_index}: residues {start_res} to {end_res}")
```

<pre>
Chain 1: residues 1 to 322
Chain 2: residues 323 to 493
Chain 3: residues 494 to 620
Chain 4: residues 621 to 729
</pre>


## Get the Second Structure

```python
from pyrosetta import init, pose_from_pdb
from pyrosetta.rosetta.core.scoring.dssp import Dssp

# 1. Initialize PyRosetta
init()
# 2. Load Your Protein Pose
pose = pose_from_pdb( "data/14-1_ImmuneCorrect.pdb")

# Run DSSP to get secondary structure
dssp = Dssp(pose)
secstruct = dssp.get_dssp_reduced_IG_as_L_secstruct()
```

<pre>
LLEEEEELELLLLLLEEEELLEEEEEELLEEELEELLLLLLEEEELLELLEELLLELHHHHHHLLLLLLLL
LLLLLLLLEEELLLLLELLLLLLLELLHHHHHHHLLLELLLEEEELLLLLLLLLLEELLLLEHLHLLLLLLE
LLLLEEEEEELLLLLLLEEEEEELLLLLLEEEEEEEEELLLHHHHHHHHLLLLLLEEEEELLLEEEELLLLL
LLLLLLLLLLEEEEEEEEELLLLEEEEEELLLEEEELEEEELLELLLLLEEELLLLEEEEEELEELLLLLEL...
</pre>

!!! note What does it mean?
    - H: Alpha-Helix
    - E: Beta-Strand
    - L: Loop or Irregular Region


<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>

