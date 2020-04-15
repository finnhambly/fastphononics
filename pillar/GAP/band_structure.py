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
npm = Atoms(symbols=(['Si'] * 128),
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

npm.set_calculator(calc)

dyn = LBFGS(atoms=npm, trajectory='npm.traj', restart='npm.pckl')
dyn.run(fmax=0.05, steps=10)

# Phonopy calculation
unitcell = PhonopyAtoms(symbols=(['Si'] * 128),
                    cell=np.diag((a, a, a)),
                    scaled_positions=npm.get_scaled_positions())
# CREATE SUPERCELL
smat = [(3, 0, 0), (0, 1, 0), (0, 0, 1)]
phonon = Phonopy(unitcell, smat, primitive_matrix='auto')
phonon.generate_displacements(distance=0.03)
phonon.save()

# CALCULATE DISPLACEMENTS
print("[Phonopy] Atomic displacements:")
disps = phonon.get_displacements()
for d in disps:
    print("[Phonopy] %d %s" % (d[0], d[1:]))
phonon.save()


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
phonon.save()

# PRODUCE FORCE CONSTANTS
phonon.produce_force_constants(forces=set_of_forces)
phonon.save()
# print('')
print("[Phonopy] Phonon frequencies at Gamma:")
# for i, freq in enumerate(phonon.get_frequencies((0, 0, 0))):
    # print("[Phonopy] %3d: %10.5f THz" %  (i + 1, freq)) # THz

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
phonon.plot_band_structure_and_dos().show()
