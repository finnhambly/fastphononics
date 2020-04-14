import os
import sys
import builtins
import ase.io.castep
import numpy as np
import ase, ase.build
from ase import Atoms
from ase.optimize import LBFGS
import quippy, quippy.descriptors
from quippy.potential import Potential

# SET UP UNIT CELL
a = 5.4307098388671875
npm = Atoms(symbols=(['Si'] * 8),
                    cell=np.diag((a, a, a)),
                    scaled_positions=[
                      (0, 0, 0),
                      (0, 0.5, 0.5),
                      (0.5, 0, 0.5),
                      (0.5, 0.5, 0),
                      (0.25, 0.25, 0.25),
                      (0.25, 0.75, 0.75),
                      (0.75, 0.25, 0.75),
                      (0.75, 0.75, 0.25)])

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
