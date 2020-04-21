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
unitcell = PhonopyAtoms(['Si'] * 127,
                    cell=np.diag((4*a, a, 7*a)),
                    scaled_positions=[
                     (0.08274653,  0.10445219,  0.18868194),
                     (0.04976774,  0.61702697,  0.26562129),
                     (0.01554113,  0.04059156,  0.256434  ),
                     (0.05359614,  0.2353898 ,  0.33464637),
                     (0.10819123,  0.919409  ,  0.37088211),
                     (0.21174374,  0.0267507 ,  0.33892984),
                     (0.24382678,  0.88843098,  0.2552962 ),
                     (0.12688912,  0.92271551,  0.26089273),
                     (0.31100254,  0.10177296,  0.20345073),
                     (0.29427828,  0.5133404 ,  0.2702335 ),
                     (0.36246624,  0.13549271,  0.28107797),
                     (0.39368756, -0.37736338,  0.3799278 ),
                     (0.32105349,  0.85173881,  0.33052644),
                     (0.28766642,  0.35642831,  0.33345983),
                     (0.39124853,  0.722797  ,  0.25473073),
                     (0.31684361,  0.66815205,  0.19125678),
                     (0.47188588, -0.3201377 ,  0.31673233),
                     (0.43093644,  0.0392345 ,  0.35541071),
                     (0.47278123,  0.04999731,  0.27040111),
                     (0.59960201,  0.16996918,  0.30663469),
                     (0.51473439,  0.87722012,  0.32160759),
                     (0.80739287,  0.22764276,  0.3282247 ),
                     (0.81175915,  0.72755905,  0.22448285),
                     (0.59675612,  0.79031428,  0.26129871),
                     (0.87222912,  0.33896222,  0.21127408),
                     (0.90387109,  0.17061173,  0.28921697),
                     (0.79045136,  0.14210758,  0.24292311),
                     (0.97875754, -0.17871226,  0.3292131 ),
                     (0.8412282 ,  0.83363621,  0.30520482),
                     (1.00174325,  0.12283402,  0.41000727),
                     (1.00972604,  0.76699044,  0.18702291),
                     (0.90884816,  0.92608993,  0.21399505),
                     (0.01431322, -0.32034777,  0.41563455),
                     (0.03670169,  0.70157459,  0.54833488),
                     (0.1234543 ,  0.41617737,  0.39664637),
                     (0.06028099,  0.36795059,  0.46546845),
                     (0.13073888,  0.79419805,  0.58613965),
                     (0.1375883 ,  0.23105155,  0.60997877),
                     (0.13527139,  1.05620792,  0.45091577),
                     (0.05938852,  0.54042475,  0.35481186),
                     (0.22499337,  0.25527567,  0.41781408),
                     (0.1749363 ,  1.07638956,  0.53348326),
                     (0.31742884,  0.46482033,  0.41330667),
                     (0.32765333,  0.3380068 ,  0.54658529),
                     (0.33930134,  0.64029144,  0.57176249),
                     (0.38870094, -0.02239731,  0.54726597),
                     (0.34236806,  1.03191287,  0.41415863),
                     (0.35382546,  0.66898194,  0.45906126),
                     (0.44637778, -0.10914451,  0.43890227),
                     (0.56638527,  0.60703579,  0.38109679),
                     (0.49526794, -0.45441786,  0.40053473),
                     (0.53988609,  0.11596441,  0.59776733),
                     (0.51503181,  1.11229033,  0.68314371),
                     (0.64694185,  0.16215061,  0.61898478),
                     (0.64732813,  0.78351165,  0.33755413),
                     (0.58178078,  0.04322482,  0.39020603),
                     (0.75636787, -0.1113686 ,  0.3595061 ),
                     (0.76953424,  0.80986689,  0.53528903),
                     (0.88663301,  0.10107297,  0.39106151),
                     (0.8063351 , -0.18533141,  0.43543454),
                     (0.84338562,  0.12675134,  0.54187901),
                     (1.02989519,  0.3364429 ,  0.60441439),
                     (0.88307653,  0.6753744 ,  0.38206134),
                     (0.78768098,  0.23422891,  0.41509429),
                     (0.01356101, -0.07251141,  0.6217281 ),
                     (0.02549124,  0.57395391,  0.72210599),
                     (0.06306347,  0.21020151,  0.68268991),
                     (0.0972875 , -0.18654119,  0.67170993),
                     (0.0473618 ,  0.95763522,  0.75844222),
                     (0.23097937,  0.20293131,  0.66893982),
                     (0.21041074,  0.79055195,  0.65632318),
                     (0.06816566,  0.11066908,  0.53530967),
                     (0.29443134,  0.00447673,  0.60285679),
                     (0.26842829,  0.53429824,  0.7082947 ),
                     (0.34433169,  0.09657203,  0.67650894),
                     (0.34658493, -0.30740752,  0.65602808),
                     (0.26289393,  0.9510414 ,  0.73774758),
                     (0.44545833, -0.08156469,  0.66600812),
                     (0.43650348,  0.63891453,  0.60260531),
                     (0.39527303,  0.2737319 ,  0.61206695),
                     (0.47962928,  0.01387846,  0.58499342),
                     (0.49207092,  0.30517987,  0.64957463),
                     (0.57221881,  0.44232895,  0.64919022),
                     (0.62309513,  0.2567562 ,  0.71120618),
                     (0.58355396,  0.86743543,  0.72564898),
                     (0.75864235, -0.05392928,  0.75010165),
                     (0.67823861,  0.89436696,  0.68166845),
                     (0.58277381,  0.79322913,  0.64177227),
                     (0.7419812 ,  0.20951772,  0.56532445),
                     (0.7599403 ,  0.85735449,  0.62215973),
                     (0.81882294,  0.22420834,  0.63114144),
                     (0.83860806, -0.08660944,  0.69316641),
                     (0.73291101,  0.26273687,  0.69471418),
                     (0.9793333 ,  0.2709192 ,  0.73411076),
                     (0.9518555 ,  0.88346374,  0.69776915),
                     (0.93252037,  0.26259439,  0.65404866)])

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

unitcell.set_calculator(calc)

# CREATE SUPERCELL
smat = [(2, 0, 0), (0, 2, 0), (0, 0, 1)]
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
