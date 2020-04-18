import os
import sys
import builtins
import numpy as np
import ase, ase.build
from ase.visualize import view
from ase import Atoms
from ase.optimize import LBFGS
import quippy, quippy.descriptors
from quippy.potential import Potential

# SET UP UNIT CELL
a = 5.431
npm = Atoms(symbols=(['Si'] * 135),
                    cell=np.diag((5*a, a, 6*a)),
                    pbc=[1, 1, 1],
                    scaled_positions=[
                (0.999999999999769,    0.999999999998788,    0.166666233639059),
                (0.999999999999740,    0.499999999998802,    0.249999708409282),
                (0.049999997056202,    0.249999999356957,    0.208332970617882),
                (0.099999994932058,    0.000000002185852,    0.249999707858053),
                (0.049999996861960,    0.750000009446396,    0.291666446612876),
                (0.149999990044168,    0.250000020796684,    0.291666443544694),
                (0.149999992516334,    0.749999994539611,    0.208332971837582),
                (0.099999995290025,    0.500000000594089,    0.166666234563123),
                (0.199999991560433,    0.000000006018182,    0.166666238699652),
                (0.199999991906846,    0.500000009188200,    0.249999709035067),
                (0.249999990217689,    0.249999990837912,    0.208332975985567),
                (0.299999992084950,    0.000000005330611,    0.249999710872860),
                (0.249999986657662,    0.750000025005743,    0.291666442862630),
                (0.349999989567150,    0.250000022900690,    0.291666442254864),
                (0.349999990887027,    0.749999998491550,    0.208332977386588),
                (0.299999989754994,    0.500000004704994,    0.166666243865902),
                (0.399999993800676,    0.000000003289983,    0.166666247693558),
                (0.399999993861853,    0.500000003151361,    0.249999709092372),
                (0.449999996929527,    0.250000001589142,    0.208332978507154),
                (0.499999999999697,    0.999999999998811,    0.249999708613209),
                (0.449999996247218,    0.750000010304964,    0.291666439120821),
                (0.550000003753450,    0.249999989697522,    0.291666439120811),
                (0.550000003071017,    0.749999998413161,    0.208332978507145),
                (0.499999999999749,    0.499999999998808,    0.166666248533873),
                (0.600000006198847,    0.999999996707651,    0.166666247693555),
                (0.600000006137623,    0.499999996846293,    0.249999709092356),
                (0.650000009113457,    0.250000001510762,    0.208332977386594),
                (0.700000007914589,    0.999999994666820,    0.249999710872860),
                (0.650000010433337,    0.749999977102230,    0.291666442254831),
                (0.750000013342795,    0.249999974997274,    0.291666442862641),
                (0.750000009782795,    0.750000009164427,    0.208332975985574),
                (0.700000010244538,    0.499999995292648,    0.166666243865916),
                (0.800000008439100,    0.999999993979453,    0.166666238699640),
                (0.800000008092709,    0.499999990809171,    0.249999709035066),
                (0.850000007484157,    0.250000005462703,    0.208332971837565),
                (0.900000005067435,    0.999999997811905,    0.249999707858059),
                (0.850000009956355,    0.749999979206017,    0.291666443544708),
                (0.950000003138581,    0.249999990555865,    0.291666446612884),
                (0.950000002944280,    0.750000000645450,    0.208332970617892),
                (0.900000004709507,    0.499999999403650,    0.166666234563128),
                (0.999999999999720,    0.999999999998827,    0.333333185711905),
                (0.999999999999672,    0.499999999998778,    0.416666663308779),
                (0.049999994642963,    0.250000013320897,    0.374999922562660),
                (0.099999989066445,    0.000000001507431,    0.416666658043720),
                (0.049999995011047,    0.750000014323485,    0.458333399348543),
                (0.149999985608437,    0.250000021182805,    0.458333390827914),
                (0.149999984922944,    0.750000022956227,    0.374999916890677),
                (0.099999991563184,    0.500000000705610,    0.333333182902816),
                (0.199999983189524,    0.000000002034894,    0.333333177957702),
                (0.199999979787738,    0.500000006627137,    0.416666649467624),
                (0.249999978046844,    0.250000026668839,    0.374999909498390),
                (0.299999977999559,    0.000000005610633,    0.416666639079472),
                (0.249999979769344,    0.750000020198666,    0.458333378744516),
                (0.349999981733220,    0.250000016559686,    0.458333365969871),
                (0.349999983709494,    0.750000021014505,    0.374999904179985),
                (0.299999983345131,    0.500000006359821,    0.333333175132105),
                (0.399999991099305,    0.000000002261228,    0.333333170862931),
                (0.399999986089826,    0.499999997410276,    0.416666631387914),
                (0.449999993917677,    0.250000012852830,    0.374999899369628),
                (0.499999999999666,    0.999999999998732,    0.416666625926960),
                (0.449999992333294,    0.750000012583223,    0.458333355342624),
                (0.550000007667341,    0.249999987419590,    0.458333355342612),
                (0.550000006082948,    0.749999987149780,    0.374999899369629),
                (0.499999999999687,    0.499999999998652,    0.333333168572301),
                (0.600000008900126,    0.999999997736070,    0.333333170862960),
                (0.600000013909542,    0.500000002586678,    0.416666631387902),
                (0.650000016291099,    0.249999978988554,    0.374999904179994),
                (0.700000021999923,    0.999999994386140,    0.416666639079491),
                (0.650000018267334,    0.749999983443262,    0.458333365969878),
                (0.750000020231146,    0.249999979804262,    0.458333378744516),
                (0.750000021953604,    0.749999973334232,    0.374999909498408),
                (0.700000016654279,    0.499999993637091,    0.333333175132100),
                (0.800000016809972,    0.999999997962040,    0.333333177957694),
                (0.800000020211767,    0.499999993369818,    0.416666649467652),
                (0.850000015077579,    0.249999977046639,    0.374999916890697),
                (0.900000010932994,    0.999999998489964,    0.416666658043709),
                (0.850000014392097,    0.749999978819783,    0.458333390827911),
                (0.950000004989609,    0.249999985678910,    0.458333399348528),
                (0.950000005357657,    0.749999986681616,    0.374999922562641),
                (0.900000008436243,    0.499999999291470,    0.333333182902802),
                (0.999999999999731,    0.999999999998847,    0.500000139918947),
                (0.999999999999755,    0.499999999998893,    0.583333611576078),
                (0.049999998156911,    0.250000015331470,    0.541666873403505),
                (0.100000001061079,    0.000000019449701,    0.583333606370790),
                (0.050000003507861,    0.750000009071957,    0.625000341455083),
                (0.150000004261712,    0.250000051168694,    0.625000333777976),
                (0.149999994140197,    0.750000026898799,    0.541666864520958),
                (0.099999992976196,    0.500000006382437,    0.500000133742446),
                (0.199999986156649,    0.000000008660386,    0.500000121397451),
                (0.200000000457349,    0.500000013262057,    0.583333590246334),
                (0.249999990977765,    0.250000021846972,    0.541666848035305),
                (0.300000001606058,    0.000000004564624,    0.583333569074168),
                (0.250000013487354,    0.750000010005509,    0.625000312627650),
                (0.350000013386588,    0.250000009654148,    0.625000290923533),
                (0.349999992224849,    0.750000014858511,    0.541666826933076),
                (0.299999983389800,    0.500000002158404,    0.500000104546921),
                (0.399999988958273,    0.000000000368099,    0.500000088670302),
                (0.400000002024744,    0.500000005133926,    0.583333549094603),
                (0.449999997562705,    0.250000006805931,    0.541666812222793),
                (0.499999999999704,    0.999999999998819,    0.583333540933919),
                (0.450000005387427,    0.750000005665189,    0.625000279242145),
                (0.549999994613136,    0.249999994336953,    0.625000279242121),
                (0.550000002437816,    0.749999993196740,    0.541666812222796),
                (0.499999999999702,    0.499999999998865,    0.500000080395319),
                (0.600000011041179,    0.999999999629157,    0.500000088670297),
                (0.599999997974784,    0.499999994863822,    0.583333549094593),
                (0.650000007775654,    0.249999985143979,    0.541666826933117),
                (0.699999998393497,    0.999999995433032,    0.583333569074173),
                (0.649999986613860,    0.749999990348234,    0.625000290923505),
                (0.749999986513112,    0.249999989996828,    0.625000312627629),
                (0.750000009022694,    0.749999978155527,    0.541666848035270),
                (0.700000016609702,    0.499999997838519,    0.500000104546930),
                (0.800000013842924,    0.999999991337140,    0.500000121397468),
                (0.799999999542154,    0.499999986735683,    0.583333590246353),
                (0.850000005860262,    0.249999973103822,    0.541666864520959),
                (0.899999998938408,    0.999999980548109,    0.583333606370787),
                (0.849999995738762,    0.749999948833546,    0.625000333777974),
                (0.949999996492581,    0.249999990930310,    0.625000341455078),
                (0.950000001843589,    0.749999984670919,    0.541666873403494),
                (0.900000007023283,    0.499999993615193,    0.500000133742436),
                (0.350000020166328,    0.750000016349137,    0.708333760648325),
                (0.300000020511710,    0.499999981648039,    0.666667034935549),
                (0.400000014930514,    0.999999999287032,    0.666667021183390),
                (0.400000015805649,    0.499999998412826,    0.750000495686606),
                (0.450000009519092,    0.250000002646012,    0.708333756224820),
                (0.499999999999694,    0.999999999998946,    0.750000495989774),
                (0.450000008436528,    0.749999996149632,    0.791667235363053),
                (0.549999991563947,    0.250000003852591,    0.791667235363055),
                (0.549999990481472,    0.749999997356478,    0.708333756224797),
                (0.499999999999725,    0.499999999998779,    0.666667015873435),
                (0.599999985068976,    0.000000000710523,    0.666667021183398),
                (0.599999984194091,    0.500000001586119,    0.750000495686827),
                (0.649999979834140,    0.249999983653045,    0.708333760648304),
                (0.699999979488072,    0.500000018350899,    0.666667034935774),
                (0.499999999999992,    0.499999999999989,    0.833333975982584)])


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

dyn = LBFGS(atoms=npm, trajectory='pillar.traj', restart='pillar.pckl')
dyn.run(fmax=0.05)
view(npm)

print(npm.get_scaled_positions())
