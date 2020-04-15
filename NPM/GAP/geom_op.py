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
ax = 3
ay = 3
az = 2
npm = Atoms(symbols=(['Si'] * 80),
                    cell=np.diag((ax*a, ay*a, az*a)),
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
                    #pillar: 2,2,2
                    (1   /ax, 1   /ay, 1   /az),
                    (1   /ax, 1.5 /ay, 1.5 /az),
                    (1.5 /ax, 1   /ay, 1.5 /az),
                    (1.5 /ax, 1.5 /ay, 1   /az),
                    (1.25/ax, 1.25/ay, 1.25/az),
                    (1.25/ax, 1.75/ay, 1.75/az),
                    (1.75/ax, 1.25/ay, 1.75/az),
                    (1.75/ax, 1.75/ay, 1.25/az)])
view(npm)

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

dyn = LBFGS(atoms=npm, trajectory='lbfgs.traj')
dyn.run(fmax=0.05)

print(npm.get_scaled_positions())
