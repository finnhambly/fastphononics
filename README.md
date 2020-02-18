# fastphononics
A program that uses Gaussian approximation potentials for finding the thermal
properties of solid state materials.

## Getting Started
An Anaconda/Miniconda installation is recommended, but is not a necessity.
### Prerequisites
Ensure GCC, gfortran, Python 3, BLAS, and LAPACK are installed:
```
sudo apt-get install gcc gfortran python python-pip libblas-dev liblapack-dev
```
If Python 2 is installed as the default when running the `python` command, make
sure that `python3` and `pip3` commands are used. Otherwise,
`update-alternatives` can be used to change the default
(`sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1`).
 Setting an alias is not sufficient.

#### ASE, NumPy, and f90wrap:
```
pip install numpy ase f90wrap # use pip3 if Python 2 is set as default
```

#### QUIP, Quippy, and GAP
Download the most recent QUIP package
```
git clone --recursive https://github.com/libAtoms/QUIP.git
```
and then download GAP from www.libatoms.org/gap/gap_download.html and place it
in the `QUIP/src` directory.
```
cd QUIP/src
tar -xzvf GAP.tar.gz
cd ..
export QUIP_ARCH=linux_x86_64_gfortran
export QUIPPY_INSTALL_OPTS=--user  # omit for a system-wide installation
make config
# press y to compile for GAP support (first additional feature)
# press enter for all other options, setting them to their default
make
make install-quippy
```
Run the following to ensure Quippy is successfully able to be loaded (nothing
  should be returned after running `import quippy`:
```
python
>>> import quippy
>>>
```

If unsuccesful, see the up-to-date documentation at
https://libatoms.github.io/GAP/

#### Install Phonopy and Phono3py
If using a conda environment, simply run
```
conda install -c conda-forge phonopy h5py
conda install -c atztogo phono3py
```
Otherwise, see https://atztogo.github.io/phonopy/install.html and
https://atztogo.github.io/phono3py/install.html

#### ShengBTE (and spglib)
Install spglib as described here: https://github.com/atztogo/spglib

Clone the source code for ShengBTE, `git clone
https://bitbucket.org/sousaw/shengbte.git`, and place a suitable `arch.make`
file in `ShengBTE/Src`. This is the `arch.make` file I used:
```
export FFLAGS=-O2 -Wall
export LDFLAGS=-L/lib -lsymspg #replace lib with path to spglib
export MPIFC=mpifort
MKL=-llapack -lblas
export LAPACK=$(MKL)
export LIBS=$(LAPACK)
```
Finally, in `shengBTE/Src`, use `make` to create ShengBTE executable.
