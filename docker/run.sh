#!/usr/bin/env bash
# Run a command inside the FIT3143 benchmark container.
#
#   docker/run.sh nproc
#   docker/run.sh make -C week-4-lab-1
#   docker/run.sh            # interactive shell
#
# The repo is bind-mounted at /work, so host edits are visible immediately and
# build artefacts land back on the host.
set -euo pipefail

IMAGE=fit3143-bench
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[run.sh] image '$IMAGE' not found, building..." >&2
    docker build -t "$IMAGE" "$REPO_ROOT/docker"
fi

# --cpus caps the container at the host's physical core count so thread counts
# in the benchmarks map to real cores. Interactive when no command is given.
CPUS="${BENCH_CPUS:-10}"

if [ $# -eq 0 ]; then
    exec docker run --rm -it --cpus="$CPUS" \
        -v "$REPO_ROOT":/work -w /work "$IMAGE"
else
    exec docker run --rm --cpus="$CPUS" \
        -v "$REPO_ROOT":/work -w /work "$IMAGE" "$@"
fi
