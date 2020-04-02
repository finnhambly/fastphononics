import numpy as np
import ase, ase.build
from ase import Atoms
from phonopy.structure.atoms import PhonopyAtoms
import quippy, quippy.descriptors
from quippy.potential import Potential
import phonopy
import phono3py
# band structure
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phono3py.file_IO import parse_FORCES_FC3
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phono3py.file_IO import (write_FORCES_FC3, write_FORCES_FC2,
    write_fc3_dat, write_fc2_dat)
from phono3py.phonon3.conductivity_RTA import get_thermal_conductivity_RTA
from phono3py.phonon3.conductivity_LBTE import get_thermal_conductivity_LBTE

filename="phonopy_params.yaml"
phonon = phono3py.load(filename)
# disp_dataset = parse_disp_fc3_yaml(filename)
# forces_fc3 = parse_FORCES_FC3(disp_dataset, filename="FORCE_CONSTANTS_3RD")
phonon.run_thermal_conductivity(
        temperatures=range(0, 1001, 10),
        boundary_mfp=1e6, # This is to avoid divergence of phonon life time.
        write_kappa=True)
