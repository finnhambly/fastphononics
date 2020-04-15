import numpy as np
import ase, ase.build
from ase import Atoms
from phonopy.structure.atoms import PhonopyAtoms
import quippy, quippy.descriptors
from quippy.potential import Potential
from phonopy import Phonopy
from phono3py import Phono3py

# SET UP UNIT CELL
a = 5.431
ax = 3
ay = 3
az = 2
unitcell = PhonopyAtoms(['Si'] * 80),
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

# CREATE SUPERCELL
smat = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
phonon = Phono3py(unitcell, smat, primitive_matrix='auto')
phonon.generate_displacements(distance=0.03)

# CALCULATE DISPLACEMENTS
print("[Phonopy] Atomic displacements:")
disp_dataset = phonon.get_displacement_dataset()
scells_with_disps = phonon.get_supercells_with_displacements()

# CALCULATE DISTANCES
count = 0
for i, disp1 in enumerate(disp_dataset['first_atoms']):
    print("%4d: %4d                %s" % (
        count + 1,
        disp1['number'] + 1,
        np.around(disp1['displacement'], decimals=3)))
    count += 1

distances = []
for i, disp1 in enumerate(disp_dataset['first_atoms']):
    for j, disp2 in enumerate(disp1['second_atoms']):
        print("%4d: %4d-%4d (%6.3f)  %s %s" % (
            count + 1,
            disp1['number'] + 1,
            disp2['number'] + 1,
            disp2['pair_distance'],
            np.around(disp1['displacement'], decimals=3),
            np.around(disp2['displacement'], decimals=3)))
        distances.append(disp2['pair_distance'])
        count += 1

# Find unique pair distances
distances = np.array(distances)
distances_int = (distances * 1e5).astype(int)
unique_distances = np.unique(distances_int) * 1e-5 # up to 5 decimals
print("Unique pair distances")
print(unique_distances)

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
    # print(("[Phonopy] Drift force:" + "%11.5f" * 3) % tuple(drift_force))
    # Simple translational invariance
    for force in forces:
        force -= drift_force / forces.shape[0]
    set_of_forces.append(forces)

# PRODUCE FORCE CONSTANTS
phonon.produce_fc3(set_of_forces, displacement_dataset=disp_dataset)
fc3 = phonon.get_fc3()
fc2 = phonon.get_fc2()

phonon._set_mesh_numbers([20, 20, 20])
print("mesh set")
phonon.run_thermal_conductivity(
        temperatures=range(300, 400, 100),
        boundary_mfp=1e6,
        is_LBTE=False,
        write_kappa=True)
cond_RTA = phonon.get_thermal_conductivity()
print(cond_RTA.get_kappa())
phonon.run_thermal_conductivity(
        temperatures=range(300, 400, 100),
        boundary_mfp=1e6,
        is_LBTE=True,
        write_kappa=True)
cond_LBTE = phonon.get_thermal_conductivity()
print(cond_LBTE.get_kappa())

qpoints = cond_RTA.get_qpoints()
print(qpoints.shape)
