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
unitcell = PhonopyAtoms(['Si'] * 96,
                    cell=np.diag((4*a, a, 5*a)),
                    pbc=[1, 1, 0],
                    scaled_positions=[
                      (9.99998780e-01 2.61822515e-05 1.98761724e-01),
                      (9.99997719e-01, 4.99994160e-01, 2.99493204e-01),
                      (6.25009970e-02, 2.49947585e-01, 2.48856400e-01),
                      (1.25001987e-01, 5.12576595e-07, 2.99501190e-01),
                      (6.24962854e-02, 7.49992782e-01, 3.49711316e-01),
                      (1.87496819e-01, 2.50017450e-01, 3.49715245e-01),
                      (1.87509725e-01, 7.50027211e-01, 2.48858764e-01),
                      (1.24894137e-01, 5.00438939e-01, 1.98787985e-01),
                      (2.50060171e-01, 9.99743820e-01, 1.98759068e-01),
                      (2.50000304e-01, 5.00014455e-01, 2.99504523e-01),
                      (3.12477394e-01, 2.50002776e-01, 2.48857127e-01),
                      (3.74995417e-01, 3.60461676e-06, 2.99493579e-01),
                      (3.12504760e-01, 7.50009088e-01, 3.49717777e-01),
                      (4.37494708e-01, 2.49982311e-01, 3.49715295e-01),
                      (4.37502410e-01, 7.49999588e-01, 2.48855519e-01),
                      (3.75102749e-01, 4.99549152e-01, 1.98789817e-01),
                      (4.99831518e-01, 6.63321078e-04, 1.98771046e-01),
                      (4.99993888e-01, 4.99979107e-01, 2.99496706e-01),
                      (5.62500632e-01, 2.49950779e-01, 2.48857173e-01),
                      (6.25002952e-01, 7.76479955e-06, 2.99503430e-01),
                      (5.62495894e-01, 7.49986560e-01, 3.49710537e-01),
                      (6.87496446e-01, 2.50032929e-01, 3.49716790e-01),
                      (6.87516832e-01, 7.50093305e-01, 2.48859642e-01),
                      (6.25055756e-01, 4.99824429e-01, 1.98794992e-01),
                      (7.50003255e-01, 9.99942678e-01, 1.98737995e-01),
                      (7.49997776e-01, 5.00012833e-01, 2.99501870e-01),
                      (8.12462472e-01, 2.49901437e-01, 2.48857815e-01),
                      (8.74992435e-01, 2.99451363e-06, 2.99497429e-01),
                      (8.12503467e-01, 7.49984701e-01, 3.49718306e-01),
                      (9.37496215e-01, 2.50003736e-01, 3.49711817e-01),
                      (9.37509082e-01, 7.50071678e-01, 2.48855083e-01),
                      (8.75040035e-01, 4.99769569e-01, 1.98812096e-01),
                      (9.99997984e-01, 9.99999511e-01, 4.00183518e-01),
                      (3.59581999e-07, 4.99998485e-01, 5.00003561e-01),
                      (6.24973011e-02, 2.50003360e-01, 4.50430218e-01),
                      (1.24997232e-01, 1.08115627e-05, 5.00002021e-01),
                      (6.25006335e-02, 7.50001977e-01, 5.49795940e-01),
                      (1.87499325e-01, 2.49996736e-01, 5.49796180e-01),
                      (1.87498530e-01, 7.50003434e-01, 4.50436414e-01),
                      (1.24998710e-01, 5.00023587e-01, 4.00187023e-01),
                      (2.49994804e-01, 9.99998288e-01, 4.00186687e-01),
                      (2.50000035e-01, 4.99998946e-01, 5.00003635e-01),
                      (3.12498455e-01, 2.49985321e-01, 4.50434880e-01),
                      (3.75000192e-01, 9.99997430e-01, 5.00002697e-01),
                      (3.12500308e-01, 7.50001768e-01, 5.49796299e-01),
                      (4.37499083e-01, 2.49995837e-01, 5.49796420e-01),
                      (4.37499244e-01, 7.50009619e-01, 4.50431986e-01),
                      (3.74998455e-01, 4.99985556e-01, 4.00189626e-01),
                      (5.00004491e-01, 2.22695215e-06, 4.00185781e-01),
                      (4.99997397e-01, 5.00009018e-01, 5.00004590e-01),
                      (5.62497414e-01, 2.50003618e-01, 4.50436617e-01),
                      (6.25000242e-01, 9.99999583e-01, 5.00003094e-01),
                      (5.62500589e-01, 7.50001808e-01, 5.49796639e-01),
                      (6.87499806e-01, 2.49999653e-01, 5.49796366e-01),
                      (6.87500964e-01, 7.49995589e-01, 4.50433877e-01),
                      (6.24993663e-01, 5.00015699e-01, 4.00189292e-01),
                      (7.49995747e-01, 9.99999847e-01, 4.00183738e-01),
                      (7.49999749e-01, 5.00003373e-01, 5.00004158e-01),
                      (8.12494631e-01, 2.49994339e-01, 4.50434398e-01),
                      (8.74999915e-01, 1.95133870e-06, 5.00002034e-01),
                      (8.12500739e-01, 7.50003383e-01, 5.49796293e-01),
                      (9.37499560e-01, 2.49998725e-01, 5.49796214e-01),
                      (9.37501335e-01, 7.49999225e-01, 4.50433812e-01),
                      (8.75000338e-01, 4.99982500e-01, 4.00191410e-01),
                      (8.76770500e-08, 9.99999727e-01, 6.00238362e-01),
                      (9.99999985e-01, 5.00000076e-01, 7.01015726e-01),
                      (6.24998963e-02, 2.50000248e-01, 6.50419031e-01),
                      (1.24999973e-01, 2.65918152e-07, 7.01015666e-01),
                      (6.25000596e-02, 7.49999891e-01, 7.51052051e-01),
                      (1.87499947e-01, 2.49999837e-01, 7.51052138e-01),
                      (1.87499940e-01, 7.49999754e-01, 6.50419101e-01),
                      (1.24999967e-01, 4.99999980e-01, 6.00238417e-01),
                      (2.49999987e-01, 9.99999984e-01, 6.00238613e-01),
                      (2.49999880e-01, 4.99999931e-01, 7.01015791e-01),
                      (3.12499913e-01, 2.50000448e-01, 6.50419168e-01),
                      (3.74999876e-01, 9.99999898e-01, 7.01015763e-01),
                      (3.12499969e-01, 7.49999758e-01, 7.51052120e-01),
                      (4.37499951e-01, 2.49999786e-01, 7.51052099e-01),
                      (4.37499951e-01, 7.49999720e-01, 6.50419234e-01),
                      (3.74999984e-01, 4.99999986e-01, 6.00238586e-01),
                      (4.99999966e-01, 9.99999966e-01, 6.00238788e-01),
                      (4.99999968e-01, 5.00000229e-01, 7.01015896e-01),
                      (5.62499898e-01, 2.50000241e-01, 6.50419304e-01),
                      (6.24999988e-01, 1.05976853e-07, 7.01015837e-01),
                      (5.62500060e-01, 7.49999886e-01, 7.51052182e-01),
                      (6.87500048e-01, 2.50000094e-01, 7.51052104e-01),
                      (6.87500067e-01, 7.50000226e-01, 6.50419246e-01),
                      (6.25000088e-01, 4.99999724e-01, 6.00238839e-01),
                      (7.50000022e-01, 1.05379817e-07, 6.00238637e-01),
                      (7.50000073e-01, 5.00000193e-01, 7.01015809e-01),
                      (8.12500000e-01, 2.50000271e-01, 6.50419169e-01),
                      (8.75000069e-01, 1.60582263e-07, 7.01015747e-01),
                      (8.12499983e-01, 7.50000525e-01, 7.51052120e-01),
                      (9.37500053e-01, 2.50000043e-01, 7.51052133e-01),
                      (9.37500080e-01, 7.50000191e-01, 6.50419090e-01),
                      (8.75000022e-01, 5.00000106e-01, 6.00238557e-01))

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
