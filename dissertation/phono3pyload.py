# SETTINGS
# N.B. THIS SCRIPT GENERATES A SEGMENTATION FAULT
#------------------------------------------------------------------------------#
# choose structure
structure = 'bulk'
# choose mode
mode = 'sw'
# choose whether to use display
gui = True
#------------------------------------------------------------------------------#
import numpy as np
import ase, ase.build
from ase import Atoms
from ase.optimize import LBFGS
from ase.visualize import view
from phonopy.structure.atoms import PhonopyAtoms
import quippy, quippy.descriptors
from quippy.potential import Potential
from phonopy import Phonopy
from phono3py import Phono3py
import phono3py
from phono3py.cui.phono3py_yaml import Phono3pyYaml
from phono3py.file_IO import parse_fc2, parse_fc3
from phonopy.harmonic.force_constants import show_drift_force_constants
from phono3py.phonon3.fc3 import show_drift_fc3

filename='phono3py_params_'+structure+mode+'.yaml'
ph3py_yaml = Phono3pyYaml()
ph3py_yaml.read(filename)
cell = ph3py_yaml.unitcell
smat = ph3py_yaml.supercell_matrix
ph_smat = ph3py_yaml.phonon_supercell_matrix
pmat = 'auto'
phonon = Phono3py(cell,
                smat,
                primitive_matrix=pmat,
                phonon_supercell_matrix=ph_smat)
num_atoms = len(phonon.unitcell.get_chemical_symbols())
fc3 = parse_fc3(num_atoms,filename='fc3'+structure+mode+'.dat')
fc2 = parse_fc2(num_atoms,filename='fc2'+structure+mode+'.dat')
phonon.set_fc3(fc3)
phonon.set_fc2(fc2)
show_drift_fc3(fc3)
show_drift_force_constants(fc2, name='fc2')
# print('[Phono3py] Setting mesh numbers:')
# # disp_dataset = parse_FORCE_SETS(filename='FORCE_SETS'+structure+mode))
#
# sw_pot = Potential('IP SW', param_str="""<SW_params n_types="3"
#     label="PRB_31_plus_H_Ge"><per_type_data type="1" atomic_num="14" />
#     <per_pair_data atnum_i="14" atnum_j="14" AA="7.049556277" BB="0.6022245584"
#     p="4" q="0" a="1.80" sigma="2.0951" eps="2.1675" />
#     <per_triplet_data atnum_c="14" atnum_j="14" atnum_k="14" lambda="21.0"
#     gamma="1.20" eps="2.1675" />
#     </SW_params>""") #from https://libatoms.github.io/GAP/quippy-potential-tutorial.html
# calc = sw_pot
#
# phonon.generate_displacements(distance=0.01)
#
# print("[Phono3py] Calculating atomic displacements")
# disp_dataset = phonon.get_displacement_dataset()
# scells_with_disps = phonon.get_supercells_with_displacements()
#
# # CALCULATE DISTANCES
# count = 0
# for i, disp1 in enumerate(disp_dataset['first_atoms']):
#     print("%4d: %4d                %s" % (
#         count + 1,
#         disp1['number'] + 1,
#         np.around(disp1['displacement'], decimals=3)))
#     count += 1
#
# distances = []
# for i, disp1 in enumerate(disp_dataset['first_atoms']):
#     for j, disp2 in enumerate(disp1['second_atoms']):
#         print("%4d: %4d-%4d (%6.3f)  %s %s" % (
#             count + 1,
#             disp1['number'] + 1,
#             disp2['number'] + 1,
#             disp2['pair_distance'],
#             np.around(disp1['displacement'], decimals=3),
#             np.around(disp2['displacement'], decimals=3)))
#         distances.append(disp2['pair_distance'])
#         count += 1
#
# # Find unique pair distances
# distances = np.array(distances)
# distances_int = (distances * 1e5).astype(int)
# unique_distances = np.unique(distances_int) * 1e-5 # up to 5 decimals
# print("Unique pair distances")
# print(unique_distances)
#
# # CALCULATE FORCES
# set_of_forces = []
# for scell in scells_with_disps:
#     cell = Atoms(symbols=scell.get_chemical_symbols(),
#                  scaled_positions=scell.get_scaled_positions(),
#                  cell=scell.get_cell(),
#                  pbc=True)
#     cell.set_calculator(calc)
#     forces = cell.get_forces()
#     drift_force = forces.sum(axis=0)
#     print(("[Phono3py] Drift force:" + "%11.5f" * 3) % tuple(drift_force))
#     # Simple translational invariance
#     for force in forces:
#         force -= drift_force / forces.shape[0]
#     set_of_forces.append(forces)


print('[Phono3py] Setting mesh numbers:')
phonon._set_mesh_numbers([10,10,10])
t = range(0, 1001, 50)
phonon.run_thermal_conductivity(
        temperatures=t,
        boundary_mfp=1e6,
        is_LBTE=True,
        write_kappa=True)
lbte = phonon.get_thermal_conductivity()
print('')
print('[Phono3py] Thermal conductivity (LBTE):')
for i in range(len(t)):
    print(("%12.3f " + "%15.7f") %
        ( lbte.get_temperatures()[i], lbte.get_kappa()[0][i][1]))

import code
code.interact(local=locals())
