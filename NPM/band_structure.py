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
ax = 3
ay = 3
az = 2
npm = Atoms(symbols=(['Si'] * 80),
                    cell=np.diag((ax*a, ay*a, az*a)),
                    pbc=[1, 1, 1],
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

npm.set_calculator(calc)

dyn = LBFGS(atoms=npm, trajectory='331+111.traj', restart='331+111.pckl')
dyn.run(fmax=0.05, steps=10)

# Phonopy calculation
unitcell = PhonopyAtoms(symbols=(['Si'] * 80),
                    cell=np.diag((a, a, a)),
                    scaled_positions=npm.get_scaled_positions())
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
phonon.save(settings={'force_constants': True, 'create_displacements': True})
labels = ["$\\Gamma$", "X", "U", "K", "$\\Gamma$", "L", "W"]
qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=51)
phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)
phonon.plot_band_structure_and_dos().show()
