# Sourced by every sbatch script in this directory. Not executable on its own.
#
# CAAS facts that shaped these settings (student-caas-headnode, Sep 2026):
#   - partition defq, 14+ nodes x 16 cores (EPYC 7763), Gigabit ethernet.
#     MaxTime is 00:20:00 (was 30 min until ~2026-09-17). A job asking for
#     more sits at PartitionTimeLimit forever rather than failing at submit,
#     so check `scontrol show partition defq` before trusting any --time.
#   - nodes may be shared with research clusters, hence median-of-3 reps
#   - Open MPI 4.1.5 is a module; mpirun cannot spawn orted on compute nodes
#     from inside an allocation. srun is the launcher -- but with NO --mpi
#     flag: --mpi=pmix and --mpi=pmi2 both hang before the step launches
#     (verified job 133637), while srun's configured default works.
#   - cluster results go to bench/results-caas/ so analyse.py's default glob
#     over bench/results/ never mixes them with the single-host Mac data.
# Pin the module the unit documents, so the MPI version behind every number is
# unambiguous; the bare names are fallbacks if the pinned one is withdrawn.
module load openmpi/4.1.5-gcc-11.2.0-ux65npg 2>/dev/null ||
  module load openmpi 2>/dev/null ||
  module load openmpi4/gcc/4.1.8 2>/dev/null ||
  echo 'WARNING: no openmpi module loaded' >&2

cd "$SLURM_SUBMIT_DIR"
export OUT_DIR="$SLURM_SUBMIT_DIR/bench/results-caas"
mkdir -p "$OUT_DIR"

# Rank count is appended by the sweep scripts. --cpu-bind=none matters for the
# hybrid (see sweep-hybrid.sh) and is harmless for Task 1.
export LAUNCH_MPI="srun -n"
export LAUNCH_HYBRID="srun --cpu-bind=none -n"
export LAUNCH_THREADS_FLAG="-c"

# mktemp -d would land in node-local /tmp, which rank 0 and the other nodes do
# not share. Point it at NFS home so every rank sees the same scratch.
export TMPDIR="$SLURM_SUBMIT_DIR/bench/slurm/scratch"
mkdir -p "$TMPDIR"

# ...but Open MPI follows TMPDIR too, and putting its shared-memory backing
# file on NFS makes intra-node shared-memory traffic go over the network --
# a silent slowdown that would land in t_compute and corrupt every speedup
# number (the smoke run, job 133661, warned about exactly this). Pin Open
# MPI's session dir to node-local /tmp; only the sweep scratch stays on NFS.
export OMPI_MCA_orte_tmpdir_base=/tmp

# Array tasks start at the same second, so the timestamped CSV names would
# collide without a per-task tag.
export RUN_TAG="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+-t$SLURM_ARRAY_TASK_ID}"

echo "job $SLURM_JOB_ID on $SLURM_JOB_NODELIST ($SLURM_JOB_NUM_NODES nodes, $SLURM_NTASKS tasks)"
echo "started $(date)"
