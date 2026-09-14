# FIT3143 Lab #2 - benchmark environment
#
# macOS cannot host this work: Apple clang rejects -fopenmp and there is no
# system Open MPI. This image provides a consistent Linux toolchain for both
# the Week 4 baselines and the Lab 2 MPI/hybrid implementations, so every
# number on a graph comes from the same compiler and libc.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        openmpi-bin \
        libopenmpi-dev \
        bc \
        coreutils \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Open MPI refuses to run as root without this. The container is throwaway and
# single-user, so allowing it is fine and avoids a UID-mapping dance on macOS.
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Docker Desktop on macOS gives the container fewer slots than physical cores,
# which makes mpirun refuse oversubscribed runs. We need those runs (the spec
# asks what happens beyond the core count), so permit them by default.
ENV OMPI_MCA_rmaps_base_oversubscribe=1

WORKDIR /work
CMD ["/bin/bash"]
