import numpy as np
import ase, ase.build
from ase.visualize import view
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from ase.optimize import LBFGS
import quippy, quippy.descriptors
from quippy.potential import Potential
from ase.build import bulk
import os
import sys
import builtins
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

a = 5.422
for i in range(48):
    print(a)
si_bulk = bulk('Si','diamond',a=a,cubic=True)
    si_bulk.set_calculator(calc)
    dyn = LBFGS(atoms=si_bulk)
    dyn.run(fmax=0.0000000001)
    a = a + 0.001

# energy is just higher for larger a
