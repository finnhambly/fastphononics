import os
import sys
import builtins
import numpy as np
import ase, ase.build
from ase.visualize import view
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from ase.optimize import LBFGS
import quippy, quippy.descriptors
from quippy.potential import Potential
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

# SET UP UNIT CELL
a = 5.431
npm = Atoms(symbols=(['Si'] * 223),
                    cell=np.diag((4*a, a, 13*a)),
                    pbc=[1, 1, 0],
                    scaled_positions=[
                    (0.000000007907487,    0.000000049183186,    0.076923010939471),
                    (0.000000007252175,    0.500000042848042,    0.115384557320585),
                    (0.062500007017273,    0.250000025436005,    0.096153785061235),
                    (0.125000012348914,    0.000000040484708,    0.115384557639863),
                    (0.062500008619428,    0.750000035339183,    0.134615329052568),
                    (0.187500011344309,    0.250000022425879,    0.134615331043419),
                    (0.187500009203079,    0.750000027122421,    0.096153784558368),
                    (0.125000006145581,    0.500000040280859,    0.076923012289296),
                    (0.250000008009178,    0.000000039321607,    0.076923012068126),
                    (0.250000008033717,    0.500000036148159,    0.115384558124963),
                    (0.312500008118346,    0.250000031221541,    0.096153785008897),
                    (0.375000004933819,    0.000000025813635,    0.115384558703323),
                    (0.312500006305207,    0.750000014489643,    0.134615332049136),
                    (0.437500001347786,    0.250000026994025,    0.134615332103900),
                    (0.437500005101812,    0.750000036646281,    0.096153784482965),
                    (0.375000008196873,    0.500000037890434,    0.076923011960524),
                    (0.500000008190354,    0.000000032278845,    0.076923010782538),
                    (0.500000001868247,    0.500000021933224,    0.115384557419168),
                    (0.562500004149391,    0.250000039227618,    0.096153783271956),
                    (0.624999998720951,    0.000000026512244,    0.115384556213176),
                    (0.562499996542404,    0.750000034210217,    0.134615330246492),
                    (0.687499994118205,    0.250000037414798,    0.134615328512666),
                    (0.687500004223638,    0.750000053700839,    0.096153782753575),
                    (0.625000008115189,    0.500000038583181,    0.076923011034128),
                    (0.750000011942929,    0.000000051097487,    0.076923011627778),
                    (0.749999997435950,    0.500000042381018,    0.115384556276054),
                    (0.812500006439093,    0.250000058306980,    0.096153784152546),
                    (0.874999999962295,    0.000000038390073,    0.115384557083643),
                    (0.812499995320190,    0.750000037711955,    0.134615329008040),
                    (0.937500003798225,    0.250000028541215,    0.134615328523588),
                    (0.937500006973052,    0.750000049632138,    0.096153784520015),
                    (0.875000011275634,    0.500000049991466,    0.076923011227608),
                    (0.000000003305179,    0.000000026262578,    0.153846101580009),
                    (0.000000000595237,    0.500000013722309,    0.192307644945436),
                    (0.062500006770737,    0.250000008458843,    0.173076873910828),
                    (0.125000006584216,    0.000000016346032,    0.192307648325364),
                    (0.062500002259976,    0.749999993253868,    0.211538418461263),
                    (0.187500005619772,    0.249999992286096,    0.211538422538160),
                    (0.187500008448227,    0.750000010577371,    0.173076876370391),
                    (0.125000007946857,    0.500000034125181,    0.153846102438359),
                    (0.250000008819589,    0.000000028260442,    0.153846104927092),
                    (0.250000005315286,    0.500000016421548,    0.192307651290747),
                    (0.312500004982197,    0.250000007656420,    0.173076879263671),
                    (0.374999999685364,    0.000000009524835,    0.192307653845862),
                    (0.312500001238247,    0.749999989309772,    0.211538426514701),
                    (0.437499994848576,    0.249999995844269,    0.211538427827788),
                    (0.437499998593629,    0.750000015270573,    0.173076880000131),
                    (0.375000002969124,    0.500000021333023,    0.153846106311605),
                    (0.499999995884422,    0.000000011724453,    0.153846104898061),
                    (0.499999991902534,    0.499999995793582,    0.192307653202390),
                    (0.562499990451291,    0.250000016190325,    0.173076878154390),
                    (0.624999986245917,    0.999999991325269,    0.192307650211769),
                    (0.562499988016867,    0.749999997825005,    0.211538425427703),
                    (0.687499982908393,    0.249999996466553,    0.211538421353939),
                    (0.687499985306926,    0.750000011041729,    0.173076875297957),
                    (0.624999990041011,    0.500000009447662,    0.153846102747073),
                    (0.749999987330788,    0.000000014628133,    0.153846101739113),
                    (0.749999983224018,    0.499999989674103,    0.192307646624984),
                    (0.812499989318709,    0.250000016714456,    0.173076872432758),
                    (0.874999991181183,    0.999999992068384,    0.192307643511835),
                    (0.812499986426747,    0.749999993846114,    0.211538416562330),
                    (0.937499994893109,    0.249999986584076,    0.211538415217921),
                    (0.937499998499378,    0.750000000756001,    0.173076871641636),
                    (0.874999994628925,    0.500000014891220,    0.153846100170169),
                    (0.999999997260240,    0.999999999013599,    0.230769189313836),
                    (0.999999988448172,    0.499999976450923,    0.269230733500480),
                    (0.062499999138408,    0.249999980528099,    0.249999963297475),
                    (0.124999993576063,    0.999999994961041,    0.269230740424825),
                    (0.062499977234650,    0.749999984215645,    0.288461507961619),
                    (0.187499988601207,    0.249999962973022,    0.288461517351082),
                    (0.187499997978794,    0.749999979189933,    0.249999969317019),
                    (0.125000003017843,    0.500000006700414,    0.230769193524460),
                    (0.250000000168323,    0.000000001408860,    0.230769198257813),
                    (0.249999991408671,    0.499999986230713,    0.269230745596162),
                    (0.312499995687280,    0.249999970193714,    0.249999973668042),
                    (0.374999992101871,    0.999999977713130,    0.269230748503852),
                    (0.312499990353301,    0.749999963003826,    0.288461521373172),
                    (0.437499990515090,    0.249999953922230,    0.288461523548212),
                    (0.437499991989763,    0.749999972803276,    0.249999975276538),
                    (0.374999995763293,    0.499999992479945,    0.230769201478183),
                    (0.499999989503662,    0.999999976877260,    0.230769201088434),
                    (0.499999989767206,    0.499999959744833,    0.269230748579798),
                    (0.562499988855846,    0.249999978120978,    0.249999973078180),
                    (0.624999990937442,    0.999999956849003,    0.269230744725260),
                    (0.562499992272475,    0.749999962152297,    0.288461520591688),
                    (0.687499992513213,    0.249999969646692,    0.288461515561291),
                    (0.687499986898520,    0.749999976637501,    0.249999968205682),
                    (0.624999985675974,    0.499999972391841,    0.230769197321370),
                    (0.749999982634464,    0.999999967488992,    0.230769191517366),
                    (0.749999989322074,    0.499999954555898,    0.269230739047437),
                    (0.812499983924018,    0.249999977710483,    0.249999961490673),
                    (0.874999989137081,    0.999999973027682,    0.269230732519447),
                    (0.812499999927769,    0.749999948005280,    0.288461507516247),
                    (0.937499985794263,    0.249999966446945,    0.288461505532285),
                    (0.937499989850656,    0.749999972846900,    0.249999958891508),
                    (0.874999987217694,    0.499999972030817,    0.230769187509423),
                    (0.249999989116469,    0.999999990213888,    0.307692293622969),
                    (0.249999997087476,    0.499999956915555,    0.346153845982506),
                    (0.312499991666187,    0.249999942034564,    0.326923070633935),
                    (0.374999995045025,    0.999999946695632,    0.346153845126016),
                    (0.312500001806138,    0.749999928948940,    0.365384620298329),
                    (0.437500000978434,    0.249999921032143,    0.365384618068603),
                    (0.437499992687208,    0.749999926686556,    0.326923071312311),
                    (0.374999988490019,    0.499999960871111,    0.307692296861881),
                    (0.499999991124793,    0.999999939450733,    0.307692296981337),
                    (0.499999997940671,    0.499999927128893,    0.346153843756877),
                    (0.562499997410714,    0.249999941561210,    0.326923068325974),
                    (0.624999997047722,    0.999999935196551,    0.346153841278612),
                    (0.562500001871756,    0.749999923167059,    0.365384616084686),
                    (0.687499999102970,    0.249999919784682,    0.365384615126123),
                    (0.687499994160026,    0.749999947906913,    0.326923065010064),
                    (0.624999993938872,    0.499999944233451,    0.307692292506376),
                    (0.250000009564717,    0.999999933746636,    0.384615396984330),
                    (0.250000006997238,    0.499999982920648,    0.423076942133375),
                    (0.312500006362634,    0.249999956670808,    0.403846168957824),
                    (0.375000013057947,    0.999999948250712,    0.423076939106118),
                    (0.312500008555691,    0.749999986903594,    0.442307714055188),
                    (0.437500010694469,    0.249999948486238,    0.442307712647119),
                    (0.437500009429215,    0.749999922098060,    0.403846165850148),
                    (0.375000006774977,    0.499999932587726,    0.384615393174544),
                    (0.500000005146772,    0.999999928989634,    0.384615391211406),
                    (0.500000012882542,    0.499999931731582,    0.423076938301432),
                    (0.562500008161305,    0.249999922083311,    0.403846163869621),
                    (0.625000015633268,    0.999999958526413,    0.423076937516533),
                    (0.562500013791456,    0.749999943549422,    0.442307709917406),
                    (0.687500019139626,    0.249999970892408,    0.442307708823791),
                    (0.687500005666585,    0.749999935653171,    0.403846162179363),
                    (0.625000003405250,    0.499999936824553,    0.384615389902179),
                    (0.250000001829438,    0.000000001999607,    0.461538484374385),
                    (0.249999994195835,    0.500000012282008,    0.500000027614326),
                    (0.312500000812848,    0.249999987631335,    0.480769257496024),
                    (0.375000004658260,    0.000000006674123,    0.500000030137787),
                    (0.312499991794409,    0.750000022416775,    0.519230800039831),
                    (0.437500002061268,    0.250000022792157,    0.519230801428440),
                    (0.437500006692588,    0.749999984290444,    0.480769258631150),
                    (0.375000009974118,    0.499999973904634,    0.461538485681108),
                    (0.500000016423032,    0.999999953070193,    0.461538485089328),
                    (0.500000012119723,    0.499999985479240,    0.500000029445950),
                    (0.562500014979331,    0.249999980721265,    0.480769256816398),
                    (0.625000023081021,    0.999999981715899,    0.500000027376110),
                    (0.562500013699878,    0.750000007874865,    0.519230799476737),
                    (0.687500026360516,    0.249999984490920,    0.519230797870957),
                    (0.687500023983894,    0.749999977813988,    0.480769254367544),
                    (0.625000020749460,    0.499999971157893,    0.461538483780616),
                    (0.249999984251246,    0.000000030104157,    0.538461570035405),
                    (0.249999979061517,    0.500000067084583,    0.576923110127499),
                    (0.312499984580285,    0.250000041036661,    0.557692340798571),
                    (0.374999990588125,    0.000000049374037,    0.576923110625107),
                    (0.312499984049964,    0.750000054841955,    0.596153880397851),
                    (0.437499996076812,    0.250000042543402,    0.596153879559296),
                    (0.437499997999007,    0.750000042473006,    0.557692341596679),
                    (0.374999995612587,    0.500000036223471,    0.538461571453847),
                    (0.500000008436285,    0.000000018941914,    0.538461571199711),
                    (0.500000004230230,    0.500000036044484,    0.576923109897683),
                    (0.562500011671387,    0.250000039738456,    0.557692339913833),
                    (0.625000018206076,    0.000000039380455,    0.576923108192543),
                    (0.562500010076687,    0.750000043791163,    0.596153877415650),
                    (0.687500022235405,    0.250000042587271,    0.596153876888210),
                    (0.687500025510812,    0.750000025963870,    0.557692338738612),
                    (0.625000022858866,    0.500000004252764,    0.538461568836750),
                    (0.249999977439490,    0.000000067605901,    0.615384650333356),
                    (0.249999978852434,    0.500000069805647,    0.653846189685667),
                    (0.312499984321356,    0.250000045649164,    0.634615420412383),
                    (0.374999991982945,    0.000000034986544,    0.653846189413701),
                    (0.312499986315675,    0.750000042143321,    0.673076960852212),
                    (0.437499999267274,    0.250000022750213,    0.673076959005474),
                    (0.437499996980099,    0.750000027065968,    0.634615419041271),
                    (0.374999989677010,    0.500000043170885,    0.615384649631557),
                    (0.500000002350130,    0.000000028820199,    0.615384648429814),
                    (0.500000004324491,    0.500000023350396,    0.653846188164772),
                    (0.562500009529629,    0.250000026480791,    0.634615417577233),
                    (0.625000016491455,    0.000000028410011,    0.653846186871653),
                    (0.562500011812125,    0.750000017927966,    0.673076957705109),
                    (0.687500022062509,    0.250000047896451,    0.673076954714563),
                    (0.687500023189509,    0.750000039650506,    0.634615415865310),
                    (0.625000014969353,    0.500000038348339,    0.615384647258660),
                    (0.249999983710564,    0.000000066470433,    0.692307731721410),
                    (0.249999996526977,    0.500000042577732,    0.730769278102512),
                    (0.312499989833763,    0.250000029321914,    0.711538503956421),
                    (0.375000001329927,    0.000000042853090,    0.730769274842215),
                    (0.312500002825593,    0.750000033635113,    0.750000048796850),
                    (0.437500001332142,    0.250000048532597,    0.750000046514865),
                    (0.437499998344478,    0.750000023066587,    0.711538501303625),
                    (0.374999993174856,    0.500000021006389,    0.692307730879687),
                    (0.500000005178328,    0.000000013902280,    0.692307729270329),
                    (0.500000006941239,    0.500000036337576,    0.730769273196267),
                    (0.562500007524384,    0.250000016866249,    0.711538500229538),
                    (0.625000007561529,    0.000000007974211,    0.730769273355959),
                    (0.562500002370255,    0.750000034088618,    0.750000046923529),
                    (0.687500000300831,    0.249999990640942,    0.750000048724048),
                    (0.687500011982774,    0.750000019135172,    0.711538499892607),
                    (0.625000014531608,    0.500000010435321,    0.692307728590373),
                    (0.250000001851973,    0.000000003964565,    0.769230824414353),
                    (0.249999998640065,    0.499999968843165,    0.807692369842637),
                    (0.312500005047627,    0.250000018438298,    0.788461594759572),
                    (0.374999999991722,    0.000000002403289,    0.807692368075339),
                    (0.312499995553885,    0.749999991725300,    0.826923140499056),
                    (0.437499998775453,    0.250000021029302,    0.826923138729858),
                    (0.437500001346381,    0.750000031266001,    0.788461593117696),
                    (0.375000003595405,    0.500000031037582,    0.769230821197716),
                    (0.499999999155966,    0.000000034990507,    0.769230819521331),
                    (0.499999999504344,    0.500000019433353,    0.807692366089598),
                    (0.562499999162574,    0.250000001072026,    0.788461593754692),
                    (0.624999998871640,    0.999999994284417,    0.807692367653541),
                    (0.562499999890983,    0.749999987277303,    0.826923140442282),
                    (0.687499998656871,    0.249999944864251,    0.826923143752369),
                    (0.687499997830402,    0.749999956395673,    0.788461596282023),
                    (0.624999997405569,    0.500000008903890,    0.769230820252738),
                    (0.249999992777385,    0.999999942255843,    0.846153917348139),
                    (0.249999993305989,    0.499999951125189,    0.884615463758346),
                    (0.312499990266714,    0.249999986331414,    0.865384686510045),
                    (0.374999994661789,    0.999999984921787,    0.884615456978815),
                    (0.312499990376338,    0.750000015825601,    0.903846231715629),
                    (0.437499988368894,    0.250000005045936,    0.903846228328163),
                    (0.437499996164972,    0.750000000652006,    0.865384683432025),
                    (0.374999993915335,    0.499999982517177,    0.846153913349935),
                    (0.499999997429479,    0.000000011434161,    0.846153911377414),
                    (0.499999993712836,    0.500000000375415,    0.884615456900052),
                    (0.562499995871669,    0.249999977437420,    0.865384686651407),
                    (0.624999994107497,    0.999999966239389,    0.884615459090580),
                    (0.562499990321778,    0.749999968676940,    0.903846231647289),
                    (0.687499997639247,    0.749999938096381,    0.865384689466363),
                    (0.624999998486902,    0.499999976279180,    0.846153914513213)])


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

dyn = LBFGS(atoms=npm, trajectory='bestwall.traj', restart='bestwall.pckl')
dyn.run(fmax=0.02)
view(npm)

print(npm.get_scaled_positions())

unitcell = PhonopyAtoms(['Si'] * 96,
                    cell=np.diag((4*a, a, 5*a)),
                    scaled_positions=npm.get_scaled_positions())


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

phonon.save(filename="phonopy_params_membrane.yaml",
settings={'force_constants': True, 'create_displacements': True})

# PLOT BAND STRUCTURE
path = [[[0.5, 0.25, 0.75], [0, 0, 0], [0.5, 0, 0.5],
        [0.5, 0.25, 0.75], [0.5, 0.5, 0.5], [0, 0, 0], [0.375, 0.375, 0.75],
        [0.5, 0.25, 0.75], [0.625, 0.25, 0.625], [0.5, 0, 0.5]]]
labels = ["$\\Gamma$", "X", "U", "K", "$\\Gamma$", "L", "W"]
qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=51)
phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)
phonon.plot_band_structure_and_dos().show()
