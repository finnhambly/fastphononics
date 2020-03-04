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
from phono3py.file_IO import parse_FORCES_FC3

# SET UP UNIT CELL
# cell = ase.build.bulk('Si', 'diamond', 5.44)
a = 5.431020511
unitcell = PhonopyAtoms(symbols=(['Si'] * 8),
                    cell=np.diag((a, a, a)),
                    scaled_positions=[(0, 0, 0),
                                      (0, 0.5, 0.5),
                                      (0.5, 0, 0.5),
                                      (0.5, 0.5, 0),
                                      (0.25, 0.25, 0.25),
                                      (0.25, 0.75, 0.75),
                                      (0.75, 0.25, 0.75),
                                      (0.75, 0.75, 0.25)])

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
primitive_matrix = [(a, 0, 0), (0, a, 0), (0, 0, a)]
phonon = Phono3py(unitcell, smat, primitive_matrix)
phonon.generate_displacements(distance=0.03)

# CALCULATE DISPLACEMENTS
print("[Phonopy] Atomic displacements:")
disps = phonon.get_displacements()
for d in disps:
    print("[Phonopy] %d %s" % (d[0], d[1:]))

# CALCULATE FORCES
supercells = phonon.get_supercells_with_displacements()
set_of_forces = []
for scell in supercells:
    cell = Atoms(symbols=scell.get_chemical_symbols(),
                 scaled_positions=scell.get_scaled_positions(),
                 cell=scell.get_cell(),
                 pbc=True)
    cell.set_calculator(calc)
    forces = cell.get_forces()
    drift_force = forces.sum(axis=0)
    print(("[Phonopy] Drift force:" + "%11.5f" * 3) % tuple(drift_force))
    # Simple translational invariance
    for force in forces:
        force -= drift_force / forces.shape[0]
    set_of_forces.append(forces)

# PRODUCE FORCE CONSTANTS
phonon.produce_force_constants(forces=set_of_forces)
print('')
print("[Phonopy] Phonon frequencies at Gamma:")
for i, freq in enumerate(phonon.get_frequencies((0, 0, 0))):
    print("[Phonopy] %3d: %10.5f THz" %  (i + 1, freq)) # THz

# DOS
phonon.set_mesh([21, 21, 21])
phonon.set_total_DOS(tetrahedron_method=True)
print('')
print("[Phonopy] Phonon DOS:")
for omega, dos in np.array(phonon.get_total_DOS()).T:
    print("%15.7f%15.7f" % (omega, dos))

# PLOT BAND STRUCTURE
path = [[[0.5, 0.25, 0.75], [0, 0, 0], [0.5, 0, 0.5],
        [0.5, 0.25, 0.75], [0.5, 0.5, 0.5], [0, 0, 0], [0.375, 0.375, 0.75],
        [0.5, 0.25, 0.75], [0.625, 0.25, 0.625], [0.5, 0, 0.5]]]
phonon.save(settings={'force_constants': True, 'create_displacements': True})
labels = ["$\\Gamma$", "X", "U", "K", "$\\Gamma$", "L", "W"]
qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=51)
phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)
# phonon.plot_band_structure_and_dos().show()
