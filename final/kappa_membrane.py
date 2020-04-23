import os
import sys
import builtins
import numpy as np
import ase, ase.build
from ase import Atoms
from ase.visualize import view
from ase.optimize import LBFGS
from phonopy.structure.atoms import PhonopyAtoms
import quippy, quippy.descriptors
from quippy.potential import Potential
from phonopy import Phonopy
from phono3py import Phono3py

# SET UP UNIT CELL
a = 5.431
npm = Atoms(symbols=(['Si'] * 96),
                    cell=np.diag((4*a, a, 5*a)),
                    pbc=[1, 1, 0],
                    scaled_positions=[
                    (0.0000,    0.00,    0.199999),
                    (0.0000,    0.50,    0.299999),
                    (0.0625,    0.25,    0.249999),
                    (0.1250,    0.00,    0.299999),
                    (0.0625,    0.75,    0.349999),
                    (0.1875,    0.25,    0.349999),
                    (0.1875,    0.75,    0.249999),
                    (0.1250,    0.50,    0.199999),
                    (0.2500,    0.00,    0.199999),
                    (0.2500,    0.50,    0.299999),
                    (0.3125,    0.25,    0.249999),
                    (0.3750,    0.00,    0.299999),
                    (0.3125,    0.75,    0.349999),
                    (0.4375,    0.25,    0.349999),
                    (0.4375,    0.75,    0.249999),
                    (0.3750,    0.50,    0.199999),
                    (0.5000,    0.00,    0.199999),
                    (0.5000,    0.50,    0.299999),
                    (0.5625,    0.25,    0.249999),
                    (0.6250,    0.00,    0.299999),
                    (0.5625,    0.75,    0.349999),
                    (0.6875,    0.25,    0.349999),
                    (0.6875,    0.75,    0.249999),
                    (0.6250,    0.50,    0.199999),
                    (0.7500,    0.00,    0.199999),
                    (0.7500,    0.50,    0.299999),
                    (0.8124,    0.25,    0.249999),
                    (0.8750,    0.00,    0.299999),
                    (0.8125,    0.75,    0.349999),
                    (0.9375,    0.25,    0.349999),
                    (0.9375,    0.75,    0.249999),
                    (0.8750,    0.50,    0.199999),
                    (0.0000,    0.00,    0.399999),
                    (0.0000,    0.50,    0.500000),
                    (0.0625,    0.25,    0.449999),
                    (0.1250,    0.00,    0.500000),
                    (0.0625,    0.75,    0.550000),
                    (0.1875,    0.25,    0.550000),
                    (0.1875,    0.75,    0.449999),
                    (0.1250,    0.50,    0.399999),
                    (0.2500,    0.00,    0.399999),
                    (0.2500,    0.50,    0.500000),
                    (0.3125,    0.25,    0.449999),
                    (0.3750,    0.00,    0.500000),
                    (0.3125,    0.75,    0.550000),
                    (0.4375,    0.25,    0.550000),
                    (0.4375,    0.75,    0.449999),
                    (0.3750,    0.50,    0.399999),
                    (0.5000,    0.00,    0.399999),
                    (0.5000,    0.50,    0.500000),
                    (0.5625,    0.25,    0.449999),
                    (0.6250,    0.00,    0.500000),
                    (0.5625,    0.75,    0.550000),
                    (0.6875,    0.25,    0.550000),
                    (0.6875,    0.75,    0.449999),
                    (0.6250,    0.50,    0.399999),
                    (0.7500,    0.00,    0.399999),
                    (0.7500,    0.50,    0.500000),
                    (0.8125,    0.25,    0.449999),
                    (0.8750,    0.00,    0.500000),
                    (0.8125,    0.75,    0.550000),
                    (0.9375,    0.25,    0.550000),
                    (0.9375,    0.75,    0.449999),
                    (0.8750,    0.50,    0.399999),
                    (0.0000,    0.00,    0.600000),
                    (0.0000,    0.50,    0.700000),
                    (0.0625,    0.25,    0.650000),
                    (0.1250,    0.00,    0.700000),
                    (0.0625,    0.75,    0.750000),
                    (0.1875,    0.25,    0.750000),
                    (0.1875,    0.75,    0.650000),
                    (0.1250,    0.50,    0.600000),
                    (0.2500,    0.00,    0.600000),
                    (0.2500,    0.50,    0.700000),
                    (0.3125,    0.25,    0.650000),
                    (0.3750,    0.00,    0.700000),
                    (0.3125,    0.75,    0.750000),
                    (0.4375,    0.25,    0.750000),
                    (0.4375,    0.75,    0.650000),
                    (0.3750,    0.50,    0.600000),
                    (0.5000,    0.00,    0.600000),
                    (0.5000,    0.50,    0.700000),
                    (0.5625,    0.25,    0.650000),
                    (0.6250,    0.00,    0.700000),
                    (0.5625,    0.75,    0.750000),
                    (0.6875,    0.25,    0.750000),
                    (0.6875,    0.75,    0.650000),
                    (0.6250,    0.50,    0.600000),
                    (0.7500,    0.00,    0.600000),
                    (0.7500,    0.50,    0.700000),
                    (0.8125,    0.25,    0.650000),
                    (0.8750,    0.00,    0.700000),
                    (0.8125,    0.75,    0.750000),
                    (0.9375,    0.25,    0.750000),
                    (0.9375,    0.75,    0.650000),
                    (0.8750,    0.50,    0.600000)])


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

dyn = LBFGS(atoms=npm, trajectory='membrane.traj', restart='membrane.pckl')
dyn.run(fmax=0.05)

print(npm.get_scaled_positions())

unitcell = PhonopyAtoms(['Si'] * 96,
                    cell=np.diag((4*a, a, 5*a)),
                    scaled_positions=npm.get_scaled_positions())

# CREATE SUPERCELL
smat = [(2, 0, 0), (0, 2, 0), (0, 0, 1)]
phonon = Phono3py(unitcell, smat, primitive_matrix='auto')
phonon.generate_displacements(distance=0.03, cutoff_pair_distance=4.0)

# CALCULATE DISPLACEMENTS
print("[Phono3py] Calculating atomic displacements")
disp_dataset = phonon.get_displacement_dataset()
scells_with_disps = phonon.get_supercells_with_displacements()

phonon.save(filename="phono3py_params_membrane.yaml")

# CALCULATE DISTANCES
# count = 0
# for i, disp1 in enumerate(disp_dataset['first_atoms']):
    # print("%4d: %4d                %s" % (
        # count + 1,
        # disp1['number'] + 1,
        # np.around(disp1['displacement'], decimals=3)))
    # count += 1

# distances = []
# for i, disp1 in enumerate(disp_dataset['first_atoms']):
#     for j, disp2 in enumerate(disp1['second_atoms']):
#         # print("%4d: %4d-%4d (%6.3f)  %s %s" % (
#         #     count + 1,
#         #     disp1['number'] + 1,
#         #     disp2['number'] + 1,
#         #     disp2['pair_distance'],
#         #     np.around(disp1['displacement'], decimals=3),
#         #     np.around(disp2['displacement'], decimals=3)))
#         distances.append(disp2['pair_distance'])
#         count += 1
#
# # Find unique pair distances
# distances = np.array(distances)
# distances_int = (distances * 1e5).astype(int)
# unique_distances = np.unique(distances_int) * 1e-5 # up to 5 decimals
# print("Unique pair distances")
# print(unique_distances)

# CALCULATE FORCES
set_of_forces = []
for scell in scells_with_disps:
    cell = Atoms(symbols=scell.get_chemical_symbols(),
                 scaled_positions=scell.get_scaled_positions(),
                 cell=scell.get_cell(),
                 pbc=True)
    cell.set_calculator(calc)
    forces = cell.get_forces()
    drift_force = forces.sum(axis=0)
    print(("[Phono3py] Drift force:" + "%11.5f" * 3) % tuple(drift_force))
    # Simple translational invariance
    for force in forces:
        force -= drift_force / forces.shape[0]
    set_of_forces.append(forces)

# PRODUCE FORCE CONSTANTS
phonon.produce_fc3(set_of_forces, displacement_dataset=disp_dataset)
phonon.save(filename="phono3py_params_membrane.yaml",
settings={'force_constants': True})
fc3 = phonon.get_fc3()
fc2 = phonon.get_fc2()

print('[Phono3py] Setting mesh numbers')
phonon._set_mesh_numbers([12, 12, 12])
phonon.run_thermal_conductivity(
        temperatures=range(300, 400, 100),
        boundary_mfp=1e6,
        is_LBTE=False,
        write_kappa=True)
cond_RTA = phonon.get_thermal_conductivity()
print('')
print('[Phono3py] Thermal conductivity (RTA):')
print(cond_RTA.get_kappa())
print('[Phono3py] Heat capacity (RTA):')
print(cond_RTA.get_mode_heat_capacities())
print('[Phono3py] Q points (RTA):')
qpoints = cond_RTA.get_qpoints()
print(qpoints.shape)

phonon.run_thermal_conductivity(
        temperatures=range(300, 400, 100),
        boundary_mfp=1e6,
        is_LBTE=True,
        write_kappa=True)
cond_LBTE = phonon.get_thermal_conductivity()
print('')
print('[Phono3py] Thermal conductivity (LBTE):')
print(cond_LBTE.get_kappa())
print('[Phono3py] Heat capacity (LBTE):')
print(cond_LBTE.get_mode_heat_capacities())
print('[Phono3py] Q points (LBTE):')
qpoints = cond_LBTE.get_qpoints()
print(qpoints.shape)
print('[Phono3py] Thermal conductivity (LBTE: RTA):')
print(cond_LBTE.get_kappa_RTA())

phonon.save(filename="phono3py_params_membrane.yaml",
settings={'force_constants': True})
