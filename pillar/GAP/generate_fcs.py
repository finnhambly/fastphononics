import numpy as np
import ase, ase.build
from ase import Atoms
from phonopy.structure.atoms import PhonopyAtoms
import quippy, quippy.descriptors
from quippy.potential import Potential
from phonopy import Phonopy
from phono3py import Phono3py
# band structure
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phono3py.file_IO import (write_FORCES_FC3, write_FORCES_FC2,
    write_fc3_dat, write_fc2_dat)

# SET UP UNIT CELL
# cell = ase.build.bulk('Si', 'diamond', 5.44)
a = 5.4307098388671875
unitcell = PhonopyAtoms(symbols=(['Si'] * 128),
                    cell=np.diag((a, a, a)),
                    scaled_positions=[
                    (0.0000,  0.0000,  5.4309),
                    (0.0000,  2.7155,  8.1464),
                    (1.3577,  1.3577,  6.7887),
                    (2.7155,  0.0000,  8.1464),
                    (1.3577,  4.0732,  9.5042),
                    (4.0732,  1.3577,  9.5042),
                    (4.0732,  4.0732,  6.7887),
                    (2.7155,  2.7155,  5.4309),
                    (5.4309,  0.0000,  5.4309),
                    (5.4309,  2.7155,  8.1464),
                    (6.7887,  1.3577,  6.7887),
                    (8.1464,  0.0000,  8.1464),
                    (6.7887,  4.0732,  9.5042),
                    (9.5042,  1.3577,  9.5042),
                    (9.5042,  4.0732,  6.7887),
                    (8.1464,  2.7155,  5.4309),
                    (10.8619,  0.0000,  5.4309),
                    (10.8619,  2.7155,  8.1464),
                    (12.2196,  1.3577,  6.7887),
                    (13.5774,  0.0000,  8.1464),
                    (12.2196,  4.0732,  9.5042),
                    (14.9351,  1.3577,  9.5042),
                    (14.9351,  4.0732,  6.7887),
                    (13.5774,  2.7155,  5.4309),
                    (16.2928,  0.0000,  5.4309),
                    (16.2928,  2.7155,  8.1464),
                    (17.6506,  1.3577,  6.7887),
                    (19.0083,  0.0000,  8.1464),
                    (17.6506,  4.0732,  9.5042),
                    (20.3661,  1.3577,  9.5042),
                    (20.3661,  4.0732,  6.7887),
                    (19.0083,  2.7155,  5.4309),
                    (21.7238,  0.0000, 10.8619),
                    (21.7238,  2.7155, 13.5774),
                    (1.3577,  1.3577, 12.2196),
                    (2.7155,  5.4309, 13.5774),
                    (1.3577,  4.0732, 14.9351),
                    (4.0732,  1.3577, 14.9351),
                    (4.0732,  4.0732, 12.2196),
                    (2.7155,  2.7155, 10.8619),
                    (5.4309,  5.4309, 10.8619),
                    (5.4309,  2.7155, 13.5774),
                    (6.7887,  1.3577, 12.2196),
                    (8.1464,  5.4309, 13.5774),
                    (6.7887,  4.0732, 14.9351),
                    (9.5042,  1.3577, 14.9351),
                    (9.5042,  4.0732, 12.2196),
                    (8.1464,  2.7155, 10.8619),
                    (10.8619,  0.0000, 10.8619),
                    (10.8619,  2.7155, 13.5774),
                    (12.2196,  1.3577, 12.2196),
                    (13.5774,  0.0000, 13.5774),
                    (12.2196,  4.0732, 14.9351),
                    (14.9351,  1.3577, 14.9351),
                    (14.9351,  4.0732, 12.2196),
                    (13.5774,  2.7155, 10.8619),
                    (16.2928,  0.0000, 10.8619),
                    (16.2928,  2.7155, 13.5774),
                    (17.6506,  1.3577, 12.2196),
                    (19.0083,  0.0000, 13.5774),
                    (17.6506,  4.0732, 14.9351),
                    (20.3661,  1.3577, 14.9351),
                    (20.3661,  4.0732, 12.2196),
                    (19.0083,  2.7155, 10.8619),
                    (21.7238,  5.4309, 16.2929),
                    (21.7238,  2.7155, 19.0083),
                    (1.3577,  1.3577, 17.6506),
                    (2.7155,  5.4309, 19.0083),
                    (1.3577,  4.0732, 20.3661),
                    (4.0732,  1.3577, 20.3661),
                    (4.0732,  4.0732, 17.6506),
                    (2.7155,  2.7155, 16.2929),
                    (5.4309,  5.4309, 16.2929),
                    (5.4309,  2.7155, 19.0083),
                    (6.7887,  1.3577, 17.6506),
                    (8.1464,  5.4309, 19.0083),
                    (6.7887,  4.0732, 20.3660),
                    (9.5042,  1.3577, 20.3660),
                    (9.5042,  4.0732, 17.6506),
                    (8.1464,  2.7155, 16.2929),
                    (10.8619,  5.4309, 16.2928),
                    (10.8619,  2.7155, 19.0083),
                    (12.2196,  1.3577, 17.6506),
                    (13.5774,  5.4309, 19.0083),
                    (12.2196,  4.0732, 20.3660),
                    (14.9351,  1.3577, 20.3660),
                    (14.9351,  4.0732, 17.6506),
                    (13.5774,  2.7155, 16.2928),
                    (16.2928,  5.4309, 16.2929),
                    (16.2928,  2.7155, 19.0083),
                    (17.6506,  1.3577, 17.6506),
                    (19.0083,  5.4309, 19.0083),
                    (17.6506,  4.0732, 20.3661),
                    (20.3661,  1.3577, 20.3661),
                    (20.3661,  4.0732, 17.6506),
                    (19.0083,  2.7155, 16.2929),
                    (5.4309,  5.4309, 21.7238),
                    (5.4309,  2.7155, 24.4392),
                    (6.7887,  1.3577, 23.0815),
                    (8.1464,  5.4309, 24.4392),
                    (6.7887,  4.0732, 25.7970),
                    (9.5042,  1.3577, 25.7970),
                    (9.5042,  4.0732, 23.0815),
                    (8.1464,  2.7155, 21.7238),
                    (10.8619,  5.4309, 21.7238),
                    (10.8619,  2.7155, 24.4392),
                    (12.2196,  1.3577, 23.0815),
                    (13.5774,  5.4309, 24.4392),
                    (12.2196,  4.0732, 25.7970),
                    (14.9351,  1.3577, 25.7970),
                    (14.9351,  4.0732, 23.0815),
                    (13.5774,  2.7155, 21.7238),
                    (5.4309,  0.0000, 27.1547),
                    (5.4309,  2.7155, 29.8702),
                    (6.7887,  1.3577, 28.5125),
                    (8.1464,  5.4309, 29.8702),
                    (6.7887,  4.0732, 31.2279),
                    (9.5042,  1.3577, 31.2279),
                    (9.5042,  4.0732, 28.5125),
                    (8.1464,  2.7155, 27.1547),
                    (10.8619,  5.4309, 27.1547),
                    (10.8619,  2.7155, 29.8702),
                    (12.2196,  1.3577, 28.5125),
                    (13.5774,  0.0000, 29.8702),
                    (12.2196,  4.0732, 31.2279),
                    (14.9351,  1.3577, 31.2279),
                    (14.9351,  4.0732, 28.5125),
                    (13.5774,  2.7155, 27.1547)])

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
# 3x1x1 supercell of conventional unit cell
smat = [(3, 0, 0), (0, 1, 0), (0, 0, 1)]
# primitive_matrix = [(4*a, 0, 0), (0, a, 0), (0, 0, 7*a)]
# phonon = Phono3py(unitcell, smat, primitive_matrix)
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


# write_FORCES_FC2(disp_dataset, forces_fc2=None, fp=None, filename="FORCES_FC2")
# write_FORCES_FC3(disp_dataset, forces_fc3=None, fp=None, filename="FORCES_FC3")

# WRITE SECOND-ORDER FORCE CONSTANTS FILE
w = open("FORCE_CONSTANTS_2ND", 'w')
w.write("%d %d \n" % (len(fc2), len(fc2)))
for i, fcs in enumerate(fc2):
    for j, fcb in enumerate(fcs):
        w.write(" %d %d\n" % (i+1, j+1))
        for vec in fcb:
            w.write("%20.14f %20.14f %20.14f\n" % tuple(vec))

# WRITE THIRD-ORDER FORCE CONSTANTS FILE
w = open("FORCE_CONSTANTS_3RD", 'w')
for i in range(fc3.shape[0]):
    for j in range(fc3.shape[1]):
        for k in range(fc3.shape[2]):
            tensor3 = fc3[i, j, k]
            w.write(" %d \n" % (k+1))
            w.write(" %d %d %d \n" % (i + 1, j + 1, k + 1))
            for dim1 in range(tensor3.shape[0]):
                for dim2 in range(tensor3.shape[1]):
                    for dim3 in range(tensor3.shape[2]):
                        w.write(" %d %d %d " % (dim1+1, dim2+1, dim3+1))
                        w.write("%20.14f \n" % tensor3[dim1,dim2,dim3])
            w.write("\n")
