import os
import sys
import builtins
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
unitcell = PhonopyAtoms(['Si'] * 127,
                    cell=np.diag((4*a, a, 7*a)),
                    pbc=[1, 1, 0],
                    scaled_positions=[
                      (0.97194963, 0.11672837, 0.156374  ),
                      (0.0057125 , 0.50572454, 0.21368449),
                      (0.06818272, 0.24992992, 0.17719818),
                      (0.12342231, 0.97863746, 0.214075  ),
                      (0.06321624, 0.74570936, 0.25294185),
                      (0.18863117, 0.24184118, 0.24820091),
                      (0.18591652, 0.72689584, 0.17668894),
                      (0.1457833 , 0.41072309, 0.13887245),
                      (0.21907627, 0.11679638, 0.15711916),
                      (0.25354171, 0.50276256, 0.21323115),
                      (0.3164697 , 0.25325408, 0.1756907 ),
                      (0.37152742, 0.97986194, 0.21207464),
                      (0.31079318, 0.7510794 , 0.25129251),
                      (0.43724311, 0.24849215, 0.24477971),
                      (0.435045  , 0.73158025, 0.17472723),
                      (0.39470795, 0.41318055, 0.13724695),
                      (0.46907583, 0.1216851 , 0.15540699),
                      (0.50397183, 0.50686068, 0.21034055),
                      (0.56764619, 0.25579009, 0.17307873),
                      (0.62372659, 0.98581128, 0.20972295),
                      (0.562909  , 0.75286419, 0.24810776),
                      (0.68888531, 0.25139958, 0.24371542),
                      (0.68855693, 0.73237646, 0.17384815),
                      (0.64859622, 0.4159217 , 0.13594499),
                      (0.72201166, 0.12197047, 0.154662  ),
                      (0.75655484, 0.51245423, 0.21092918),
                      (0.81939525, 0.25492434, 0.17460011),
                      (0.87533116, 0.98642413, 0.21174149),
                      (0.81472042, 0.7537534 , 0.25003215),
                      (0.93968252, 0.2445701 , 0.24739424),
                      (0.93914755, 0.7278215 , 0.17594645),
                      (0.89975353, 0.41416036, 0.13748606),
                      (0.99921377, 0.99920124, 0.28682697),
                      (0.00345132, 0.48532414, 0.35979992),
                      (0.06434552, 0.24052135, 0.32246491),
                      (0.12273125, 0.98728451, 0.36060805),
                      (0.0605324 , 0.73859178, 0.39928587),
                      (0.18833051, 0.2275975 , 0.39683896),
                      (0.18674356, 0.73950348, 0.32417632),
                      (0.12765434, 0.48744548, 0.28636004),
                      (0.24799192, 0.99670246, 0.28729001),
                      (0.25097833, 0.48715773, 0.35918399),
                      (0.31351388, 0.25054964, 0.32092421),
                      (0.37926899, 0.00456467, 0.35499728),
                      (0.31731658, 0.75408369, 0.39008755),
                      (0.44275269, 0.24688742, 0.39241933),
                      (0.43857018, 0.74843172, 0.31836831),
                      (0.37601389, 0.49232143, 0.28337023),
                      (0.49916588, 0.00462345, 0.282192  ),
                      (0.50096624, 0.49647369, 0.35388761),
                      (0.56332293, 0.25352333, 0.31700941),
                      (0.62566144, 0.00228421, 0.35218427),
                      (0.56322412, 0.75571089, 0.38768563),
                      (0.68380319, 0.2591906 , 0.38890092),
                      (0.6895029 , 0.75204125, 0.31777442),
                      (0.62780681, 0.49840142, 0.28154396),
                      (0.74977763, 0.01199467, 0.28239936),
                      (0.75055165, 0.50839051, 0.35536704),
                      (0.81249516, 0.25862311, 0.31863316),
                      (0.87318875, 0.00667784, 0.3548029 ),
                      (0.81187146, 0.76667388, 0.39142574),
                      (0.93601655, 0.23418757, 0.39315126),
                      (0.93942879, 0.74049558, 0.32337488),
                      (0.8789246 , 0.49957385, 0.28438562),
                      (0.96745421, 0.89197184, 0.42876163),
                      (0.86335533, 0.44140631, 0.48353087),
                      (0.14174554, 0.54843699, 0.49212945),
                      (0.1125473 , 0.58394041, 0.53280089),
                      (0.01694544, 0.69496771, 0.55589722),
                      (0.12289481, 0.00435407, 0.53287289),
                      (0.23462332, 0.62222741, 0.46412058),
                      (0.12864421, 0.48709201, 0.43225006),
                      (0.25748373, 0.97550694, 0.43049744),
                      (0.19410629, 0.29295087, 0.52126608),
                      (0.30900884, 0.29317521, 0.46231781),
                      (0.40547875, 0.13714032, 0.48407777),
                      (0.34099323, 0.84874255, 0.53219535),
                      (0.4153305 , 0.13133482, 0.54980443),
                      (0.44240129, 0.7508626 , 0.46346424),
                      (0.37938698, 0.51155823, 0.42591955),
                      (0.50848541, 0.99452742, 0.42752898),
                      (0.4842244 , 0.41423059, 0.49598401),
                      (0.57168483, 0.24464269, 0.46450404),
                      (0.67043378, 0.1007198 , 0.49021799),
                      (0.59403878, 0.76958773, 0.53062651),
                      (0.65859673, 0.04176713, 0.56346419),
                      (0.69567179, 0.73083609, 0.45878637),
                      (0.62253686, 0.5153033 , 0.42503696),
                      (0.74856015, 0.03330869, 0.4258252 ),
                      (0.77167389, 0.53713144, 0.51768801),
                      (0.75476824, 0.36200702, 0.46928171),
                      (0.86694134, 0.4766235 , 0.52271194),
                      (0.85055026, 0.99480033, 0.57918092),
                      (0.96707363, 0.39943746, 0.52328963),
                      (0.90688034, 0.87428958, 0.48270203),
                      (0.88872009, 0.55974238, 0.42474482),
                      (0.25465004, 0.03838869, 0.56072942),
                      (0.26975725, 0.3547594 , 0.66153093),
                      (0.30761146, 0.27755059, 0.60451653),
                      (0.37450358, 0.99447245, 0.63651367),
                      (0.31450518, 0.73753004, 0.6740022 ),
                      (0.43622523, 0.22597388, 0.67566485),
                      (0.44612647, 0.73882182, 0.60373089),
                      (0.37672155, 0.51701311, 0.56892968),
                      (0.51360748, 0.97114449, 0.56675938),
                      (0.49970174, 0.47933383, 0.64220052),
                      (0.56433435, 0.24531248, 0.60635731),
                      (0.62869079, 0.96536653, 0.63864074),
                      (0.56443326, 0.7269991 , 0.67635018),
                      (0.65277501, 0.09537365, 0.69590307),
                      (0.70322929, 0.75102745, 0.60355724),
                      (0.63516204, 0.47075707, 0.57237801),
                      (0.24596981, 0.92076391, 0.71515997),
                      (0.2809556 , 0.60743202, 0.75574427),
                      (0.31230087, 0.2038638 , 0.74278004),
                      (0.38391399, 0.05460289, 0.78469031),
                      (0.35237341, 0.65492077, 0.80059752),
                      (0.42128617, 0.35891595, 0.82475957),
                      (0.4399302 , 0.76892786, 0.74784058),
                      (0.37815555, 0.49240518, 0.71183761),
                      (0.50533049, 0.99906461, 0.71134814),
                      (0.49941293, 0.53859165, 0.78742783),
                      (0.55838746, 0.28275608, 0.74874964),
                      (0.59699047, 0.89499848, 0.76735106),
                      (0.54641058, 0.85200476, 0.81911529),
                      (0.69047409, 0.81824145, 0.7390754 ),
                      (0.62903258, 0.50104382, 0.71311762)])

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
smat = [(2, 0, 0), (0, 2, 0), (0, 0, 1)]
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
    print(("[Phonopy] Drift force:" + "%11.5f" * 3) % tuple(drift_force))
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
