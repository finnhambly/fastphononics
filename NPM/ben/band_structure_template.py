import os
import sys
import builtins
import numpy as np
import ase, ase.build
from ase import Atoms
from phonopy.structure.atoms import PhonopyAtoms
from ase.optimize import LBFGS
import quippy, quippy.descriptors
from quippy.potential import Potential
from phonopy import Phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

# SET UP UNIT CELL
# cell = ase.build.bulk('Si', 'diamond', 5.44)
a = 5.431
unitcell = PhonopyAtoms(['Si'] * ?,
                    cell=np.diag((?*a, a, ?*a)),
                    pbc=[1, 1, 0],
                    scaled_positions=[])

# SET UP CALCULATOR
# Gaussian Approximation Potentials (GAP)
# Do I need this here?
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
phonon = Phonopy(unitcell, smat, primitive_matrix='auto')
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
phonon.set_mesh([12, 12, 12])
phonon.set_total_DOS(tetrahedron_method=True)
print('')
print("[Phonopy] Phonon DOS:")
for omega, dos in np.array(phonon.get_total_DOS()).T:
    print("%15.7f%15.7f" % (omega, dos))

# PLOT BAND STRUCTURE
path = [[[0.5, 0.25, 0.75], [0, 0, 0], [0.5, 0, 0.5],
        [0.5, 0.25, 0.75], [0.5, 0.5, 0.5], [0, 0, 0], [0.375, 0.375, 0.75],
        [0.5, 0.25, 0.75], [0.625, 0.25, 0.625], [0.5, 0, 0.5]]]
phonon.save(filename="phonopy_params_bestwall.yaml", settings={'force_constants': True, 'create_displacements': True})
labels = ["$\\Gamma$", "X", "U", "K", "$\\Gamma$", "L", "W"]
qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=51)
phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)
phonon.plot_band_structure_and_dos().show()
