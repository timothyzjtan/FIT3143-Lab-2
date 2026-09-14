# FIT3143 Lab 2 - MPI and hybrid MPI+OpenMP implementations.
# Build inside the container:  docker/run.sh make -C lab-2
MPICC   ?= mpicc
CFLAGS  ?= -O2 -Wall -Wextra
LDLIBS  := -lm

all: task1 task2

task1: task1.c
	$(MPICC) $(CFLAGS) $< -o $@ $(LDLIBS)

task2: task2.c
	$(MPICC) $(CFLAGS) -fopenmp $< -o $@ $(LDLIBS)

clean:
	rm -f task1 task2 *_output.txt

.PHONY: all clean
