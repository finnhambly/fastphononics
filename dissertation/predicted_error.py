#------------------------------------------------------------------------------#
structure = 'bestwall'
mode = 'gap'
#------------------------------------------------------------------------------#
from quippy.potential import Potential
from ase import Atoms
import numpy as np
import os
import sys
import builtins
import ase
from ase.io.trajectory import Trajectory
from ase.visualize import view

orig_dir = os.getcwd()
model_dir = os.path.dirname(sys.argv[0])
if model_dir != '':
    os.chdir(model_dir)

if os.path.exists('gp_iter6_sparse9k.xml.sparseX.GAP_2017_6_17_60_4_3_56_1651.bz2'):
    os.system('bunzip2 gp_iter6_sparse9k.xml.sparseX.GAP_2017_6_17_60_4_3_56_1651.bz2')
try:
    p = Potential(init_args='Potential xml_label="GAP_2017_6_17_60_4_3_56_165"',
                                               param_filename='gp_iter6_sparse9k.xml')
    p.__str__ = lambda self: '<GAP Potential>'
finally:
    os.chdir(orig_dir)
# p = Potential("",param_filename="/mnt/c/Users/Finn/Documents/fastphononics/final/gp_iter6_sparse9k.xml")
# p = Potential("",param_filename="/users/fjbh500/scratch/fastphononics/final/gp_iter6_sparse9k.xml")
p.calc_args="local_gap_variance"
a = 5.431
npm = ase.io.read(structure+'.xyz')
npm.set_calculator(p)
npm.get_potential_energy()
predicted_error = np.sqrt(npm.arrays["local_gap_variance"])
print(np.amax(predicted_error))
npm.set_initial_charges(predicted_error*100)
view(npm)

# optimised structure
traj = Trajectory(structure+mode+'.traj')
length=len(traj)-1
opt = traj[length]
opt.set_calculator(p)
opt.get_potential_energy()
predicted_error = np.sqrt(opt.arrays["local_gap_variance"])
opt.set_initial_charges(predicted_error*100)
view(opt)

import code
code.interact(local=locals())
