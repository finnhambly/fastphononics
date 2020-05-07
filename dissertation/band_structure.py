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

# SETTINGS
#------------------------------------------------------------------------------#
# choose structure
structure = 'membrane'
# choose mode
mode = 'sw'
# choose whether to use display
gui = True
# set band structure labels
latex_labels = True
#------------------------------------------------------------------------------#

# import unit cell
lattice = ase.io.read(structure+'.xyz')

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

# set interatomic potential
lattice.set_calculator(calc)

# calculate the predicted error
if mode == 'gap':
    calc.calc_args="local_gap_variance"
    lattice.get_potential_energy()
    predicted_error = np.sqrt(lattice.arrays["local_gap_variance"])
    print("Maximum force RMSE", np.amax(predicted_error))
    lattice.set_initial_charges(predicted_error*100)
    if (gui):
        view(lattice) # set colours by initial charge
# structure optimisation
dyn = LBFGS(atoms=lattice, trajectory=structure+mode+'.traj',
    restart=structure+mode+'.pckl')
dyn.run(fmax=0.01)

# calculate the predicted error after the optimisation
if mode == 'gap':
    lattice.get_potential_energy()
    predicted_error = np.sqrt(lattice.arrays["local_gap_variance"])
    lattice.set_initial_charges(predicted_error*100)
    if (gui):
        view(lattice) # set colours by initial charge

# create object for phonopy calculations
unitcell = PhonopyAtoms(lattice.get_chemical_symbols(),
                    cell=lattice.get_cell(),
                    scaled_positions=lattice.get_scaled_positions())
# create supercell
if structure == 'bulk':
    smat = [(2, 0, 0), (0, 2, 0), (0, 0, 2)]
else:
    smat = [(2, 0, 0), (0, 2, 0), (0, 0, 1)]
phonon = Phonopy(unitcell, smat, primitive_matrix='auto')
phonon.generate_displacements(distance=0.03)
phonon.save(filename='phonopy_params_'+structure+mode+'.yaml')
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
path = [[[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0]]] # path through 2D
qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=51)
if (latex_labels):
    labels = ["$\\Gamma$", "X", "L", "$\\Gamma$"]
    phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)
else:
        phonon.run_band_structure(qpoints, path_connections=connections)
if (gui):
    phonon.plot_band_structure_and_dos().show()

# Thermal properties
print('Number of Q points is:')
mesh_dict = phonon.get_mesh_dict()
qpoints_dim = mesh_dict['qpoints']
print(qpoints_dim.shape)
qpoints = qpoints_dim.shape[0]
print(qpoints)
phonon.run_thermal_properties(t_step=50,
                              t_max=1001,
                              t_min=0)
tp_dict = phonon.get_thermal_properties_dict()
temperatures = tp_dict['temperatures']
free_energy = tp_dict['free_energy']
entropy = tp_dict['entropy']
heat_capacity = tp_dict['heat_capacity']

# write animation
phonon.write_animation([qpoints,qpoints,qpoints], anime_type='v_sim',
    filename='anime_'+structure+mode+'.ascii')

for t, F, S, cv in zip(temperatures, free_energy, entropy, heat_capacity):
    print(("%12.3f " + "%15.7f" * 3) % ( t, F, S, cv ))

if (gui):
    phonon.plot_thermal_properties().show()
