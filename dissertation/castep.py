export CASTEP_COMMAND=castep.mpi
export CASTEP_PP_PATH=/opt/apps/easybuild/software/phys/CASTEP/19.1.1-foss-2018b/bin

import numpy as np
import ase, ase.build
from ase.visualize import view
import ase
import ase.calculators.castep
import ase.io.castep
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from ase.optimize import LBFGS, BFGSLineSearch
from ase.optimize.precon import PreconLBFGS, Exp
import quippy, quippy.descriptors
from quippy.potential import Potential
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

# SETTINGS
#------------------------------------------------------------------------------#
# choose structure
structure = 'bulk'
# choose mode
mode = 'castep'
# choose whether to use display
gui = False
# set band structure labels
latex_labels = False
#------------------------------------------------------------------------------#

npm = ase.io.read(structure+'.xyz')

calc = ase.calculators.castep.Castep()
directory = 'CASTEP_SI_BULK'

# include interface settings in .param file
calc._export_settings = True

# reuse the same directory
calc._directory = directory
calc._rename_existing_dir = False
calc._label = 'SI_BULK'

# necessary for tasks with changing positions
# such as GeometryOptimization or npmecularDynamics
calc._set_atoms = True

# Param settings
calc.param.xc_functional = 'PBE'
calc.param.cut_off_energy = 400
# Prevent CASTEP from writing *wvfn* files
calc.param.num_dump_cycles = 0

# Cell settings
calc.cell.kpoint_mp_grid = '6 6 6'
calc.cell.fix_com = False
calc.cell.fix_all_cell = True

# all of the following are identical
calc.param.task = 'GeometryOptimization'

# Prepare atoms
npm.set_calculator(calc)

# Check for correct input
if calc.dryrun_ok():
    print('%s : %s ' % (npm.calc._label, npm.get_potential_energy()))
else:
    print("Found error in input")
    print(calc._error)

# create object for phonopy calculations
unitcell = PhonopyAtoms(npm.get_chemical_symbols(),
                    cell=npm.get_cell(),
                    scaled_positions=npm.get_scaled_positions())
# create supercell
if structure == 'bulk':
    smat = [(2, 0, 0), (0, 2, 0), (0, 0, 2)]
else:
    smat = [(2, 0, 0), (0, 2, 0), (0, 0, 1)]
phonon = Phonopy(unitcell, smat, primitive_matrix='auto')
phonon.generate_displacements(distance=0.03)
phonon.save(filename='phonopy_params_'+structure+mode+'.yaml'),
# calculate displacements
print("[Phonopy] Atomic displacements:")
disps = phonon.get_displacements()
for d in disps:
    print("[Phonopy] %d %s" % (d[0], d[1:]))

# calculate forces
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

# produce force constants
phonon.produce_force_constants(forces=set_of_forces)

# print out the phonon frequencies
print('')
print("[Phonopy] Phonon frequencies at Gamma:")
for i, freq in enumerate(phonon.get_frequencies((0, 0, 0))):
    print("[Phonopy] %3d: %10.5f THz" %  (i + 1, freq)) # THz
print('')
print("[Phonopy] Phonon frequencies at U:")
for i, freq in enumerate(phonon.get_frequencies((0.5, 0.5, 0))):
    print("[Phonopy] %3d: %10.5f THz" %  (i + 1, freq)) # THz

# density of states
phonon.set_mesh([6, 6, 6])
phonon.set_total_DOS(tetrahedron_method=True)
print('')
print("[Phonopy] Phonon DOS:")
for omega, dos in np.array(phonon.get_total_DOS()).T:
    print("%15.7f%15.7f" % (omega, dos))

phonon.save(filename="phonopy_params_"+structure+mode+".yaml",
    settings={'force_constants': True, 'create_displacements': True})

# plot band structure
path = [[[0, 0, 0], [0, 0.5, 0.5], [0.25, 0.75, 0.5], [0, 0, 0], [0.5, 0.5, 0.5]]]
qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=51)
if (latex_labels):
    labels = ["$\\Gamma$", "X", "K", "$\\Gamma$", "L"]
    phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)
else:
        phonon.run_band_structure(qpoints, path_connections=connections)
if (gui):
    phonon.plot_band_structure_and_dos().show()
