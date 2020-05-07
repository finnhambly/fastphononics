#------------------------------------------------------------------------------#
structure = 'membrane'
mode = 'sw'
gui = True
latex_labels = True
#------------------------------------------------------------------------------#
import numpy as np
import ase, ase.build
from ase import Atoms
from phonopy.structure.atoms import PhonopyAtoms
import quippy, quippy.descriptors
from quippy.potential import Potential
from ase.visualize import view
import phonopy
import phono3py
# band structure
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phono3py.file_IO import parse_FORCES_FC3
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phono3py.file_IO import (write_FORCES_FC3, write_FORCES_FC2,
    write_fc3_dat, write_fc2_dat)
from phono3py.phonon3.conductivity_RTA import get_thermal_conductivity_RTA
from phono3py.phonon3.conductivity_LBTE import get_thermal_conductivity_LBTE

filename='phonopy_params_'+structure+mode+'.yaml'
phonon = phonopy.load(filename)

phonon.set_mesh([6, 6, 6])
phonon.set_total_DOS(tetrahedron_method=True)
print('')
print("[Phonopy] Phonon DOS:")
for omega, dos in np.array(phonon.get_total_DOS()).T:
    print("%15.7f%15.7f" % (omega, dos))

# plot band structure
path = [[[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0]]]
qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=51)
if (latex_labels):
    labels = ["$\\Gamma$", "X", "L", "$\\Gamma$"]
    phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)
else:
        phonon.run_band_structure(qpoints, path_connections=connections)
if (gui):
    phonon.plot_band_structure_and_dos().show()

# Thermal properties
print('Number of Q points is:')
mesh_dict = phonon.get_mesh_dict()
qpoints_dim = mesh_dict['qpoints']
print(qpoints_dim.shape)
qpoints = qpoints_dim.shape[0]
print(qpoints)
phonon.run_thermal_properties(t_step=100,
                              t_max=1001,
                              t_min=0)
tp_dict = phonon.get_thermal_properties_dict()
temperatures = tp_dict['temperatures']
free_energy = tp_dict['free_energy']
entropy = tp_dict['entropy']
heat_capacity = tp_dict['heat_capacity']

# write animation
phonon.write_animation([qpoints,qpoints,qpoints], anime_type='v_sim',
    filename='anime_'+structure+mode+'.ascii')

for t, F, S, cv in zip(temperatures, free_energy, entropy, heat_capacity):
    print(("%12.3f " + "%15.7f" * 3) % ( t, F, S, cv ))

if (gui):
    phonon.plot_thermal_properties().show()
    unitcell = phonon.get_unitcell()
    npm = Atoms(unitcell.get_chemical_symbols(),
        cell=unitcell.get_cell(),
        scaled_positions=unitcell.get_scaled_positions())
    view(npm)

import code
code.interact(local=locals())
