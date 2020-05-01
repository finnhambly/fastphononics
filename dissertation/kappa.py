# SETTINGS
#------------------------------------------------------------------------------#
# choose structure
structure = 'thickmembrane'
# choose mode
mode = 'gap'
# choose whether to use display
gui = False
# choose whether force constants should be written for ShengBTE
shengbte = True
#------------------------------------------------------------------------------#
import numpy as np
import ase, ase.build
from ase import Atoms
from ase.optimize import LBFGS
from ase.visualize import view
from phonopy.structure.atoms import PhonopyAtoms
import quippy, quippy.descriptors
from quippy.potential import Potential
from phonopy import Phonopy
from phono3py import Phono3py
from phono3py.cui.phono3py_yaml import Phono3pyYaml
from phono3py.file_IO import write_fc3_dat, write_fc2_dat
from phonopy.file_IO import write_FORCE_SETS

# import unit cell
npm = ase.io.read(structure+'.xyz')

# set up calculator
if mode == 'gap':
    import os
    import sys
    import builtins
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
elif mode == 'sw':
    sw_pot = Potential('IP SW', param_str="""<SW_params n_types="3"
        label="PRB_31_plus_H_Ge"><per_type_data type="1" atomic_num="14" />
        <per_pair_data atnum_i="14" atnum_j="14" AA="7.049556277" BB="0.6022245584"
        p="4" q="0" a="1.80" sigma="2.0951" eps="2.1675" />
        <per_triplet_data atnum_c="14" atnum_j="14" atnum_k="14" lambda="21.0"
        gamma="1.20" eps="2.1675" />
        </SW_params>""") #from https://libatoms.github.io/GAP/quippy-potential-tutorial.html
    calc = sw_pot
else:
    print('Please select a valid mode')

npm.set_calculator(calc)

# structure optimisation
dyn = LBFGS(atoms=npm, trajectory=structure+mode+'.traj',
    restart=structure+mode+'.pckl')
dyn.run(fmax=0.001)
if (gui):
    view(npm)

# create object for phonopy calculations
unitcell = PhonopyAtoms(npm.get_chemical_symbols(),
                    cell=npm.get_cell(),
                    scaled_positions=npm.get_scaled_positions())
# create supercell
if structure == 'bulk':
    smat = [(2, 0, 0), (0, 2, 0), (0, 0, 2)]
else:
    smat = [(2, 0, 0), (0, 2, 0), (0, 0, 1)]
phonon = Phono3py(unitcell, smat, primitive_matrix='auto')
phonon.generate_displacements(distance=0.01)

# CALCULATE DISPLACEMENTS
print("[Phono3py] Calculating atomic displacements")
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
    print(("[Phono3py] Drift force:" + "%11.5f" * 3) % tuple(drift_force))
    # Simple translational invariance
    for force in forces:
        force -= drift_force / forces.shape[0]
    set_of_forces.append(forces)
write_FORCE_SETS(disp_dataset,filename='FORCE_SETS_'+structure+mode)

# PRODUCE FORCE CONSTANTS
phonon.produce_fc3(set_of_forces, displacement_dataset=disp_dataset)
fc3 = phonon.get_fc3()
fc2 = phonon.get_fc2()

print('[Phono3py] Setting mesh numbers:')
phonon._set_mesh_numbers([10,10,10])
t = range(0, 1001, 50)
phonon.run_thermal_conductivity(
        temperatures=t,
        boundary_mfp=1e6,
        is_LBTE=True,
        write_kappa=True)
lbte = phonon.get_thermal_conductivity()
print('')
print('[Phono3py] Thermal conductivity (LBTE):')
for i in range(len(t)):
    print(("%12.3f " + "%15.7f") %
        ( lbte.get_temperatures()[i], lbte.get_kappa()[0][i][1]))

print('[Phono3py] Q points (LBTE):')
qpoints = lbte.get_qpoints()
print(qpoints.shape)

print('[Phono3py] Thermal conductivity (LBTE: RTA):')
for i in range(len(t)):
    print(("%12.3f " + "%15.7f") %
        ( lbte.get_temperatures()[i], lbte.get_kappa_RTA()[0][i][1]))

# save data
ph3py = Phono3pyYaml()
phonon.calculator = 'crystal'
ph3py.set_phonon_info(phonon)
filename='phono3py_params_'+structure+mode+'.yaml'
with open(filename, 'w') as w:
	w.write(str(ph3py))

write_fc3_dat(fc3, filename='fc3'+structure+mode+'.dat')
write_fc2_dat(fc2, filename='fc2'+structure+mode+'.dat')

# write force constants to ShengBTE input files
if (shengbte):
    # write second-order force constants file
    w = open("FORCE_CONSTANTS_2ND", 'w')
    w.write("%d %d \n" % (len(fc2), len(fc2)))
    for i, fcs in enumerate(fc2):
        for j, fcb in enumerate(fcs):
            w.write(" %d %d\n" % (i+1, j+1))
            for vec in fcb:
                w.write("%20.14f %20.14f %20.14f\n" % tuple(vec))

    # write third-order force constants file
    w = open("FORCE_CONSTANTS_3RD", 'w')
    for i in range(fc3.shape[0]):
        for j in range(fc3.shape[1]):
            for k in range(fc3.shape[2]):
                tensor3 = fc3[i, j, k]
                w.write(" %d \n" % (k+1))
                w.write(" %d %d %d \n" % (i + 1, j + 1, k + 1))
                for dim1 in range(tensor3.shape[0]):
                    for dim2 in range(tensor3.shape[1]):
                        for dim3 in range(tensor3.shape[2]):
                            w.write(" %d %d %d " % (dim1+1, dim2+1, dim3+1))
                            w.write("%20.14f \n" % tensor3[dim1,dim2,dim3])
                w.write("\n")
