# Week 4 baselines, rebuilt as the measurement instrument for Lab 2.
# Build inside the container:  docker/run.sh make -C week-4-lab-1
CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra
LDLIBS  := -lm

all: task1 task2 task3

task1: task1.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

task2: task2.c
	$(CC) $(CFLAGS) $< -o $@ -lpthread $(LDLIBS)

task3: task3.c
	$(CC) $(CFLAGS) -fopenmp $< -o $@ $(LDLIBS)

clean:
	rm -f task1 task2 task3 *_primes_output.txt

.PHONY: all clean
