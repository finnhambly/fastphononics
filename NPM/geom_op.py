import os
import sys
import builtins
import numpy as np
import ase, ase.build
from ase.visualize import view
from ase import Atoms
from ase.optimize import LBFGS
import quippy, quippy.descriptors
from quippy.potential import Potential

# SET UP UNIT CELL
a = 5.431
ax = 6
ay = 6
az = 1
unitcell =Atoms(['Si'] * 288,
                    cell=np.diag((ax*a, ay*a, az*a)),
                    pbc=[1, 1, 1],
                    scaled_positions=[
                    #1,1,1
                    (0/ax   , 0/ay   , 0/az   ),
                    (0/ax   , 0.5/ay , 0.5/az ),
                    (0.5/ax , 0/ay   , 0.5/az ),
                    (0.5/ax , 0.5/ay , 0/az   ),
                    (0.25/ax, 0.25/ay, 0.25/az),
                    (0.25/ax, 0.75/ay, 0.75/az),
                    (0.75/ax, 0.25/ay, 0.75/az),
                    (0.75/ax, 0.75/ay, 0.25/az),
                    #2,1,1
                    (1/ax   , 0/ay   , 0/az  ),
                    (1/ax   , 0.5/ay , 0.5/az),
                    (1.5/ax , 0/ay   , 0.5/az),
                    (1.5/ax , 0.5/ay , 0/az  ),
                    (1.75/ax, 0.75/ay, 0.25/az),
                    (1.25/ax, 0.25/ay, 0.25/az),
                    (1.25/ax, 0.75/ay, 0.75/az),
                    (1.75/ax, 0.25/ay, 0.75/az),
                    #3,1,1
                    (2   /ax, 0   /ay, 0   /az),
                    (2   /ax, 0.5 /ay, 0.5 /az),
                    (2.5 /ax, 0   /ay, 0.5 /az),
                    (2.5 /ax, 0.5 /ay, 0   /az),
                    (2.25/ax, 0.25/ay, 0.25/az),
                    (2.25/ax, 0.75/ay, 0.75/az),
                    (2.75/ax, 0.25/ay, 0.75/az),
                    (2.75/ax, 0.75/ay, 0.25/az),
                    #4,1,1
                    (3   /ax, 0   /ay, 0   /az),
                    (3   /ax, 0.5 /ay, 0.5 /az),
                    (3.5 /ax, 0   /ay, 0.5 /az),
                    (3.5 /ax, 0.5 /ay, 0   /az),
                    (3.25/ax, 0.25/ay, 0.25/az),
                    (3.25/ax, 0.75/ay, 0.75/az),
                    (3.75/ax, 0.25/ay, 0.75/az),
                    (3.75/ax, 0.75/ay, 0.25/az),
                    #5,1,1
                    (4   /ax, 0   /ay, 0   /az),
                    (4   /ax, 0.5 /ay, 0.5 /az),
                    (4.5 /ax, 0   /ay, 0.5 /az),
                    (4.5 /ax, 0.5 /ay, 0   /az),
                    (4.25/ax, 0.25/ay, 0.25/az),
                    (4.25/ax, 0.75/ay, 0.75/az),
                    (4.75/ax, 0.25/ay, 0.75/az),
                    (4.75/ax, 0.75/ay, 0.25/az),
                    #6,1,1
                    (5   /ax, 0   /ay, 0   /az),
                    (5   /ax, 0.5 /ay, 0.5 /az),
                    (5.5 /ax, 0   /ay, 0.5 /az),
                    (5.5 /ax, 0.5 /ay, 0   /az),
                    (5.25/ax, 0.25/ay, 0.25/az),
                    (5.25/ax, 0.75/ay, 0.75/az),
                    (5.75/ax, 0.25/ay, 0.75/az),
                    (5.75/ax, 0.75/ay, 0.25/az),
                    #1,2,1
                    (0   /ax, 1   /ay, 0   /az),
                    (0   /ax, 1.5 /ay, 0.5 /az),
                    (0.5 /ax, 1   /ay, 0.5 /az),
                    (0.5 /ax, 1.5 /ay, 0   /az),
                    (0.25/ax, 1.25/ay, 0.25/az),
                    (0.25/ax, 1.75/ay, 0.75/az),
                    (0.75/ax, 1.25/ay, 0.75/az),
                    (0.75/ax, 1.75/ay, 0.25/az),
                    #1,3,1
                    (0   /ax, 2   /ay, 0   /az),
                    (0   /ax, 2.5 /ay, 0.5 /az),
                    (0.5 /ax, 2   /ay, 0.5 /az),
                    (0.5 /ax, 2.5 /ay, 0   /az),
                    (0.25/ax, 2.25/ay, 0.25/az),
                    (0.25/ax, 2.75/ay, 0.75/az),
                    (0.75/ax, 2.25/ay, 0.75/az),
                    (0.75/ax, 2.75/ay, 0.25/az),
                    #1,4,1
                    (0   /ax, 3   /ay, 0   /az),
                    (0   /ax, 3.5 /ay, 0.5 /az),
                    (0.5 /ax, 3   /ay, 0.5 /az),
                    (0.5 /ax, 3.5 /ay, 0   /az),
                    (0.25/ax, 3.25/ay, 0.25/az),
                    (0.25/ax, 3.75/ay, 0.75/az),
                    (0.75/ax, 3.25/ay, 0.75/az),
                    (0.75/ax, 3.75/ay, 0.25/az),
                    #1,5,1
                    (0   /ax, 4   /ay, 0   /az),
                    (0   /ax, 4.5 /ay, 0.5 /az),
                    (0.5 /ax, 4   /ay, 0.5 /az),
                    (0.5 /ax, 4.5 /ay, 0   /az),
                    (0.25/ax, 4.25/ay, 0.25/az),
                    (0.25/ax, 4.75/ay, 0.75/az),
                    (0.75/ax, 4.25/ay, 0.75/az),
                    (0.75/ax, 4.75/ay, 0.25/az),
                    #1,6,1
                    (0   /ax, 5   /ay, 0   /az),
                    (0   /ax, 5.5 /ay, 0.5 /az),
                    (0.5 /ax, 5   /ay, 0.5 /az),
                    (0.5 /ax, 5.5 /ay, 0   /az),
                    (0.25/ax, 5.25/ay, 0.25/az),
                    (0.25/ax, 5.75/ay, 0.75/az),
                    (0.75/ax, 5.25/ay, 0.75/az),
                    (0.75/ax, 5.75/ay, 0.25/az),
                    #2,2,1
                    (1   /ax, 1   /ay, 0   /az),
                    (1   /ax, 1.5 /ay, 0.5 /az),
                    (1.5 /ax, 1   /ay, 0.5 /az),
                    (1.5 /ax, 1.5 /ay, 0   /az),
                    (1.25/ax, 1.25/ay, 0.25/az),
                    (1.25/ax, 1.75/ay, 0.75/az),
                    (1.75/ax, 1.25/ay, 0.75/az),
                    (1.75/ax, 1.75/ay, 0.25/az),
                    #2,3,1
                    (1   /ax, 2   /ay, 0   /az),
                    (1   /ax, 2.5 /ay, 0.5 /az),
                    (1.5 /ax, 2   /ay, 0.5 /az),
                    (1.5 /ax, 2.5 /ay, 0   /az),
                    (1.25/ax, 2.25/ay, 0.25/az),
                    (1.25/ax, 2.75/ay, 0.75/az),
                    (1.75/ax, 2.25/ay, 0.75/az),
                    (1.75/ax, 2.75/ay, 0.25/az),
                    #2,4,1
                    (1   /ax, 3   /ay, 0   /az),
                    (1   /ax, 3.5 /ay, 0.5 /az),
                    (1.5 /ax, 3   /ay, 0.5 /az),
                    (1.5 /ax, 3.5 /ay, 0   /az),
                    (1.25/ax, 3.25/ay, 0.25/az),
                    (1.25/ax, 3.75/ay, 0.75/az),
                    (1.75/ax, 3.25/ay, 0.75/az),
                    (1.75/ax, 3.75/ay, 0.25/az),
                    #2,5,1
                    (1   /ax, 4   /ay, 0   /az),
                    (1   /ax, 4.5 /ay, 0.5 /az),
                    (1.5 /ax, 4   /ay, 0.5 /az),
                    (1.5 /ax, 4.5 /ay, 0   /az),
                    (1.25/ax, 4.25/ay, 0.25/az),
                    (1.25/ax, 4.75/ay, 0.75/az),
                    (1.75/ax, 4.25/ay, 0.75/az),
                    (1.75/ax, 4.75/ay, 0.25/az),
                    #2,6,1
                    (1   /ax, 5   /ay, 0   /az),
                    (1   /ax, 5.5 /ay, 0.5 /az),
                    (1.5 /ax, 5   /ay, 0.5 /az),
                    (1.5 /ax, 5.5 /ay, 0   /az),
                    (1.25/ax, 5.25/ay, 0.25/az),
                    (1.25/ax, 5.75/ay, 0.75/az),
                    (1.75/ax, 5.25/ay, 0.75/az),
                    (1.75/ax, 5.75/ay, 0.25/az),
                    #3,2,1
                    (2   /ax, 1   /ay, 0   /az),
                    (2   /ax, 1.5 /ay, 0.5 /az),
                    (2.5 /ax, 1   /ay, 0.5 /az),
                    (2.5 /ax, 1.5 /ay, 0   /az),
                    (2.25/ax, 1.25/ay, 0.25/az),
                    (2.25/ax, 1.75/ay, 0.75/az),
                    (2.75/ax, 1.25/ay, 0.75/az),
                    (2.75/ax, 1.75/ay, 0.25/az),
                    #3,3,1
                    (2   /ax, 2   /ay, 0   /az),
                    (2   /ax, 2.5 /ay, 0.5 /az),
                    (2.5 /ax, 2   /ay, 0.5 /az),
                    (2.5 /ax, 2.5 /ay, 0   /az),
                    (2.25/ax, 2.25/ay, 0.25/az),
                    (2.25/ax, 2.75/ay, 0.75/az),
                    (2.75/ax, 2.25/ay, 0.75/az),
                    (2.75/ax, 2.75/ay, 0.25/az),
                    #3,4,1
                    (2   /ax, 3   /ay, 0   /az),
                    (2   /ax, 3.5 /ay, 0.5 /az),
                    (2.5 /ax, 3   /ay, 0.5 /az),
                    (2.5 /ax, 3.5 /ay, 0   /az),
                    (2.25/ax, 3.25/ay, 0.25/az),
                    (2.25/ax, 3.75/ay, 0.75/az),
                    (2.75/ax, 3.25/ay, 0.75/az),
                    (2.75/ax, 3.75/ay, 0.25/az),
                    #3,5,1
                    (2   /ax, 4   /ay, 0   /az),
                    (2   /ax, 4.5 /ay, 0.5 /az),
                    (2.5 /ax, 4   /ay, 0.5 /az),
                    (2.5 /ax, 4.5 /ay, 0   /az),
                    (2.25/ax, 4.25/ay, 0.25/az),
                    (2.25/ax, 4.75/ay, 0.75/az),
                    (2.75/ax, 4.25/ay, 0.75/az),
                    (2.75/ax, 4.75/ay, 0.25/az),
                    #3,6,1
                    (2   /ax, 5   /ay, 0   /az),
                    (2   /ax, 5.5 /ay, 0.5 /az),
                    (2.5 /ax, 5   /ay, 0.5 /az),
                    (2.5 /ax, 5.5 /ay, 0   /az),
                    (2.25/ax, 5.25/ay, 0.25/az),
                    (2.25/ax, 5.75/ay, 0.75/az),
                    (2.75/ax, 5.25/ay, 0.75/az),
                    (2.75/ax, 5.75/ay, 0.25/az),
                    #4,2,1
                    (3   /ax, 1   /ay, 0   /az),
                    (3   /ax, 1.5 /ay, 0.5 /az),
                    (3.5 /ax, 1   /ay, 0.5 /az),
                    (3.5 /ax, 1.5 /ay, 0   /az),
                    (3.25/ax, 1.25/ay, 0.25/az),
                    (3.25/ax, 1.75/ay, 0.75/az),
                    (3.75/ax, 1.25/ay, 0.75/az),
                    (3.75/ax, 1.75/ay, 0.25/az),
                    #4,3,1
                    (3   /ax, 2   /ay, 0   /az),
                    (3   /ax, 2.5 /ay, 0.5 /az),
                    (3.5 /ax, 2   /ay, 0.5 /az),
                    (3.5 /ax, 2.5 /ay, 0   /az),
                    (3.25/ax, 2.25/ay, 0.25/az),
                    (3.25/ax, 2.75/ay, 0.75/az),
                    (3.75/ax, 2.25/ay, 0.75/az),
                    (3.75/ax, 2.75/ay, 0.25/az),
                    #4,4,1
                    (3   /ax, 3   /ay, 0   /az),
                    (3   /ax, 3.5 /ay, 0.5 /az),
                    (3.5 /ax, 3   /ay, 0.5 /az),
                    (3.5 /ax, 3.5 /ay, 0   /az),
                    (3.25/ax, 3.25/ay, 0.25/az),
                    (3.25/ax, 3.75/ay, 0.75/az),
                    (3.75/ax, 3.25/ay, 0.75/az),
                    (3.75/ax, 3.75/ay, 0.25/az),
                    #4,5,1
                    (3   /ax, 4   /ay, 0   /az),
                    (3   /ax, 4.5 /ay, 0.5 /az),
                    (3.5 /ax, 4   /ay, 0.5 /az),
                    (3.5 /ax, 4.5 /ay, 0   /az),
                    (3.25/ax, 4.25/ay, 0.25/az),
                    (3.25/ax, 4.75/ay, 0.75/az),
                    (3.75/ax, 4.25/ay, 0.75/az),
                    (3.75/ax, 4.75/ay, 0.25/az),
                    #4,6,1
                    (3   /ax, 5   /ay, 0   /az),
                    (3   /ax, 5.5 /ay, 0.5 /az),
                    (3.5 /ax, 5   /ay, 0.5 /az),
                    (3.5 /ax, 5.5 /ay, 0   /az),
                    (3.25/ax, 5.25/ay, 0.25/az),
                    (3.25/ax, 5.75/ay, 0.75/az),
                    (3.75/ax, 5.25/ay, 0.75/az),
                    (3.75/ax, 5.75/ay, 0.25/az),
                    #5,2,1
                    (4   /ax, 1   /ay, 0   /az),
                    (4   /ax, 1.5 /ay, 0.5 /az),
                    (4.5 /ax, 1   /ay, 0.5 /az),
                    (4.5 /ax, 1.5 /ay, 0   /az),
                    (4.25/ax, 1.25/ay, 0.25/az),
                    (4.25/ax, 1.75/ay, 0.75/az),
                    (4.75/ax, 1.25/ay, 0.75/az),
                    (4.75/ax, 1.75/ay, 0.25/az),
                    #5,3,1
                    (5   /ax, 2   /ay, 0   /az),
                    (5   /ax, 2.5 /ay, 0.5 /az),
                    (5.5 /ax, 2   /ay, 0.5 /az),
                    (5.5 /ax, 2.5 /ay, 0   /az),
                    (5.25/ax, 2.25/ay, 0.25/az),
                    (5.25/ax, 2.75/ay, 0.75/az),
                    (5.75/ax, 2.25/ay, 0.75/az),
                    (5.75/ax, 2.75/ay, 0.25/az),
                    #5,4,1
                    (4   /ax, 3   /ay, 0   /az),
                    (4   /ax, 3.5 /ay, 0.5 /az),
                    (4.5 /ax, 3   /ay, 0.5 /az),
                    (4.5 /ax, 3.5 /ay, 0   /az),
                    (4.25/ax, 3.25/ay, 0.25/az),
                    (4.25/ax, 3.75/ay, 0.75/az),
                    (4.75/ax, 3.25/ay, 0.75/az),
                    (4.75/ax, 3.75/ay, 0.25/az),
                    #5,5,1
                    (4   /ax, 4   /ay, 0   /az),
                    (4   /ax, 4.5 /ay, 0.5 /az),
                    (4.5 /ax, 4   /ay, 0.5 /az),
                    (4.5 /ax, 4.5 /ay, 0   /az),
                    (4.25/ax, 4.25/ay, 0.25/az),
                    (4.25/ax, 4.75/ay, 0.75/az),
                    (4.75/ax, 4.25/ay, 0.75/az),
                    (4.75/ax, 4.75/ay, 0.25/az),
                    #5,6,1
                    (4   /ax, 0   /ay, 0   /az),
                    (4   /ax, 0.5 /ay, 0.5 /az),
                    (4.5 /ax, 0   /ay, 0.5 /az),
                    (4.5 /ax, 0.5 /ay, 0   /az),
                    (4.25/ax, 0.25/ay, 0.25/az),
                    (4.25/ax, 0.75/ay, 0.75/az),
                    (4.75/ax, 0.25/ay, 0.75/az),
                    (4.75/ax, 0.75/ay, 0.25/az),
                    #6,2,1
                    (5   /ax, 1   /ay, 0   /az),
                    (5   /ax, 1.5 /ay, 0.5 /az),
                    (5.5 /ax, 1   /ay, 0.5 /az),
                    (5.5 /ax, 1.5 /ay, 0   /az),
                    (5.25/ax, 1.25/ay, 0.25/az),
                    (5.25/ax, 1.75/ay, 0.75/az),
                    (5.75/ax, 1.25/ay, 0.75/az),
                    (5.75/ax, 1.75/ay, 0.25/az),
                    #6,3,1
                    (5   /ax, 2   /ay, 0   /az),
                    (5   /ax, 2.5 /ay, 0.5 /az),
                    (5.5 /ax, 2   /ay, 0.5 /az),
                    (5.5 /ax, 2.5 /ay, 0   /az),
                    (5.25/ax, 2.25/ay, 0.25/az),
                    (5.25/ax, 2.75/ay, 0.75/az),
                    (5.75/ax, 2.25/ay, 0.75/az),
                    (5.75/ax, 2.75/ay, 0.25/az),
                    #6,4,1
                    (5   /ax, 3   /ay, 0   /az),
                    (5   /ax, 3.5 /ay, 0.5 /az),
                    (5.5 /ax, 3   /ay, 0.5 /az),
                    (5.5 /ax, 3.5 /ay, 0   /az),
                    (5.25/ax, 3.25/ay, 0.25/az),
                    (5.25/ax, 3.75/ay, 0.75/az),
                    (5.75/ax, 3.25/ay, 0.75/az),
                    (5.75/ax, 3.75/ay, 0.25/az),
                    #6,5,1
                    (5   /ax, 4.5 /ay, 0.5 /az),
                    (5   /ax, 4   /ay, 0   /az),
                    (5.5 /ax, 4   /ay, 0.5 /az),
                    (5.5 /ax, 4.5 /ay, 0   /az),
                    (5.25/ax, 4.25/ay, 0.25/az),
                    (5.25/ax, 4.75/ay, 0.75/az),
                    (5.75/ax, 4.25/ay, 0.75/az),
                    (5.75/ax, 4.75/ay, 0.25/az),
                    #6,6,1
                    (5   /ax, 5   /ay, 0   /az),
                    (5   /ax, 5.5 /ay, 0.5 /az),
                    (5.5 /ax, 5   /ay, 0.5 /az),
                    (5.5 /ax, 5.5 /ay, 0   /az),
                    (5.25/ax, 5.25/ay, 0.25/az),
                    (5.25/ax, 5.75/ay, 0.75/az),
                    (5.75/ax, 5.25/ay, 0.75/az),
                    (5.75/ax, 5.75/ay, 0.25/az)])


# SET UP CALCULATOR
# Gaussian Approximation Potentials (GAP)
orig_dir = os.getcwd()
model_dir = os.path.dirname(sys.argv[0])
if model_dir != '':
    os.chdir(model_dir)

if os.path.exists('gp_iter6_sparse9k.xml.sparseX.GAP_2017_6_17_60_4_3_56_1651.bz2'):
    os.system('bunzip2 gp_iter6_sparse9k.xml.sparseX.GAP_2017_6_17_60_4_3_56_1651.bz2')

try:
    calc = Potential(init_args='Potential xml_label="GAP_2017_6_17_60_4_3_56_165"',
                                               param_filename='gp_iter6_sparse9k.xml')
    Potential.__str__ = lambda self: '<GAP Potential>'
finally:
    os.chdir(orig_dir)


no_checkpoint = True

npm.set_calculator(calc)

dyn = LBFGS(atoms=npm, trajectory='661.traj', restart='661.pckl')
dyn.run(fmax=0.05)
view(npm)

print(npm.get_scaled_positions())
