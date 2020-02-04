export FFLAGS=-O2 -Wall
export LDFLAGS=-L/lib -lsymspg
export MPIFC=mpifort
MKL=-llapack -lblas
export LAPACK=$(MKL)
export LIBS=$(LAPACK)
