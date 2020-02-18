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
a = 3.868
unitcell = PhonopyAtoms(symbols=(['Si'] * 2),
                    cell=np.diag((a, a, a)),
                    scaled_positions=[(0, 0, 0),
                                      (0.25, 0.25, 0.25)])

# SET UP CALCULATOR
# Stillinger-Weber potential
sw_pot = Potential('IP SW', param_str="""<SW_params n_types="3"
    label="PRB_31_plus_H_Ge"><per_type_data type="1" atomic_num="14" />
    <per_pair_data atnum_i="14" atnum_j="14" AA="7.049556277" BB="0.6022245584"
    p="4" q="0" a="1.80" sigma="2.0951" eps="2.1675" />
    <per_triplet_data atnum_c="14" atnum_j="14" atnum_k="14" lambda="21.0"
    gamma="1.20" eps="2.1675" />
    </SW_params>""") #from https://libatoms.github.io/GAP/quippy-potential-tutorial.html
calc = sw_pot

# CREATE SUPERCELL
# 2x2x2 supercell of conventional unit cell
smat = [(2, 0, 0), (0, 2, 0), (0, 0, 2)]
# primitive_matrix = [(0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
phonon = Phono3py(unitcell, smat, primitive_matrix=None)
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


# import code
# code.interact(local=locals())
