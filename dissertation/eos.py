import os
import sys
import builtins
import numpy as np
import numpy as np
import ase, ase.build
from ase.visualize import view
from ase import Atoms
import quippy
from quippy.potential import Potential
from ase.build import bulk
from ase.io.trajectory import Trajectory
from ase.io import read
from ase.units import kJ
from ase.eos import EquationOfState

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

a = 5.45  # approximate lattice constant
si_bulk = bulk('Si','diamond',a=a,cubic=True)
si_bulk.set_calculator(calc)
cell = si_bulk.get_cell()
traj = Trajectory('si_bulk.traj', 'w')
for x in np.linspace(0.9, 1.1, 20):
    si_bulk.set_cell(cell * x, scale_atoms=True)
    si_bulk.get_potential_energy()
    traj.write(si_bulk)

configs = read('si_bulk.traj@0:20')  # read 20 configurations
# Extract volumes and energies:
volumes = [si_bulk.get_volume() for si_bulk in configs]
energies = [si_bulk.get_potential_energy() for si_bulk in configs]
eos = EquationOfState(volumes, energies)
v0, e0, B = eos.fit()
print(B / kJ * 1.0e24, 'GPa')
eos.plot('si_bulk-eos.png')
