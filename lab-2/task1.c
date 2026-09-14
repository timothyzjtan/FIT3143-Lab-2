/*
 * FIT3143 Lab 2 - Task 1: Prime Search with the Message Passing Interface
 *
 * Finds every prime strictly less than n across a set of MPI processes and
 * writes the sorted result to a text file from the root process.
 *
 * The trial-division kernel in is_prime() is IDENTICAL to the Week 4 serial,
 * POSIX-thread and OpenMP implementations. Speedup is T_serial / T_parallel,
 * so holding the per-candidate work constant is what makes the comparison
 * measure parallelism rather than a change of algorithm.
 *
 * ---------------------------------------------------------------------------
 * PARTITIONING SCHEMES (--scheme)
 *
 * Trial division costs ~sqrt(k) iterations, so the cost of a candidate grows
 * with its value. Any scheme that hands out equal-sized contiguous ranges
 * therefore hands the high ranks more work. Four schemes are implemented so
 * the trade-off can be measured rather than asserted:
 *
 *   block        Rank r takes one contiguous range. Simplest, and the result
 *                arrives at the root already in ascending order, so NO SORT is
 *                needed. But it is load-imbalanced: measured on the Week 4
 *                baselines at n=1e7 over 8 blocks, the slowest block did 1.6x
 *                the work of the fastest (~80% efficiency).
 *
 *   cyclic       Rank r takes k = 2+r, 2+r+P, 2+r+2P, ... Interleaving spreads
 *                the cost gradient almost perfectly, but each rank's primes are
 *                scattered through [2,n), so the root MUST sort after gathering.
 *
 *   blockcyclic  Chunks of --chunk candidates dealt round-robin to ranks.
 *                Keeps most of cyclic's balance while restoring cache locality
 *                (a rank walks contiguous memory within a chunk). Also needs a
 *                sort at the root.
 *
 *   dynamic      Master-worker: rank 0 hands chunks to workers on request, so
 *                a rank that finishes early immediately gets more. The best
 *                balance available, and the only scheme that adapts to cores of
 *                DIFFERENT SPEEDS - which matters on heterogeneous CPUs such as
 *                Apple silicon (performance + efficiency cores), where equal
 *                work on unequal cores is imbalanced by construction. Costs one
 *                rank to coordination and adds request/reply messages.
 *
 * Sorting is performed ONLY when the scheme requires it, and is timed
 * separately as t_sort. Charging `block` for a sort it does not need would
 * misrepresent its true total cost; this way each scheme is compared on what
 * it actually has to do.
 *
 * ---------------------------------------------------------------------------
 * TIMING MODEL (all via MPI_Wtime, reported by the root)
 *
 *   t_alloc    local buffer allocation                       serial
 *   t_bcast    disseminating n to all ranks                  serial
 *   t_compute  the distributed prime search                  PARALLEL
 *   t_comm     MPI_Gather + MPI_Gatherv of the results       serial
 *   t_sort     ordering the gathered result (0 for block)    serial
 *   t_io       writing the output file                       serial
 *   t_total    whole run, MPI_Init to just before MPI_Finalize
 *
 * t_compute is reduced with MPI_MAX, not averaged: the wall-clock cost of a
 * parallel phase is set by its slowest rank, so MAX is the figure that belongs
 * in a speedup calculation. The gap between MAX and MIN is the load imbalance,
 * and is reported so it can be quantified rather than guessed at.
 *
 * t_total is the figure used for speedup graphs and deliberately includes
 * communication, sorting and file writing, because the rubric grades "overall
 * speedup (including communication, computation, sorting and file writing)".
 *
 * Compile: mpicc -O2 -Wall -Wextra task1.c -o task1 -lm
 * Run:     mpirun -np 8 ./task1 10000000
 *          mpirun -np 8 ./task1 10000000 --scheme blockcyclic --chunk 4096
 *          mpirun -np 8 ./task1 10000000 --scheme dynamic --csv
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <mpi.h>

/* Message tags for the master-worker scheme. */
#define TAG_REQUEST 1
#define TAG_WORK    2

typedef enum { SCHEME_BLOCK, SCHEME_CYCLIC, SCHEME_BLOCKCYCLIC, SCHEME_DYNAMIC }
    scheme_t;

/* Trial division up to sqrt(k): if k = m * d with m <= d then m <= sqrt(k), so
 * any factor beyond sqrt(k) would already have been caught by its smaller
 * pair. Frozen across all Lab 1 and Lab 2 implementations - see header. */
static int is_prime(long k) {
    if (k < 2) return 0;
    if (k == 2) return 1;
    if (k % 2 == 0) return 0;

    long limit = (long) sqrt((double) k);
    for (long i = 3; i <= limit; i += 2) {
        if (k % i == 0) return 0;
    }
    return 1;
}

/* Upper bound on pi(n): pi(x) < 1.25506 x / ln(x) for x > 1
 * (Rosser & Schoenfeld, 1962); 1.26 with a small additive margin is safe.
 * At n = 1e7 this reserves ~782k slots for the 664,579 primes that exist,
 * rather than the n slots a flat malloc(n) would take. */
static long prime_count_bound(long n) {
    if (n < 100) return 100;
    return (long) (1.26 * (double) n / log((double) n)) + 16;
}

/* Growable result buffer. Each scheme seeds it with a good estimate, so in
 * practice growth almost never fires; it exists so that no scheme can ever
 * overrun its buffer, including `dynamic` where a rank's share is not known
 * until the run is over. Doubling keeps this amortised O(1) per append. */
typedef struct {
    long *data;
    long  count;
    long  cap;
} vec_t;

static int vec_init(vec_t *v, long cap) {
    if (cap < 64) cap = 64;
    v->data = malloc(sizeof(long) * (size_t) cap);
    v->count = 0;
    v->cap = cap;
    return v->data != NULL;
}

static int vec_push(vec_t *v, long value) {
    if (v->count == v->cap) {
        long ncap = v->cap * 2;
        long *nd = realloc(v->data, sizeof(long) * (size_t) ncap);
        if (nd == NULL) return 0;
        v->data = nd;
        v->cap = ncap;
    }
    v->data[v->count++] = value;
    return 1;
}

static int cmp_long(const void *a, const void *b) {
    long x = *(const long *) a, y = *(const long *) b;
    return (x > y) - (x < y);
}

/* Scan [start, end) and append every prime found. */
static int scan_range(vec_t *v, long start, long end) {
    for (long k = start; k < end; k++) {
        if (is_prime(k) && !vec_push(v, k)) return 0;
    }
    return 1;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: mpirun -np P %s <n> [options]\n"
        "  <n>              find primes strictly less than n\n"
        "  --scheme KIND    block | cyclic | blockcyclic | dynamic"
                          "   (default blockcyclic)\n"
        "  --chunk N        chunk size for blockcyclic/dynamic (default 4096)\n"
        "  --csv            emit one machine-readable CSV row instead of prose\n"
        "  --out FILE       output file (default task1_mpi_output.txt)\n",
        prog);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    double t_total_start = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = 0;
    long chunk = 4096;
    int csv_mode = 0;
    scheme_t scheme = SCHEME_BLOCKCYCLIC;
    const char *scheme_name = "blockcyclic";
    const char *out_path = "task1_mpi_output.txt";

    /* Only the root parses the command line; n is then broadcast. Every rank
     * sees the same argv under mpirun, but parsing centrally and broadcasting
     * is what the specification asks for and is what a real launcher would
     * need if ranks were started differently. */
    int parse_error = 0;
    if (rank == 0) {
        int have_n = 0;
        for (int i = 1; i < argc && !parse_error; i++) {
            if (strcmp(argv[i], "--csv") == 0) {
                csv_mode = 1;
            } else if (strcmp(argv[i], "--out") == 0) {
                if (i + 1 >= argc) { parse_error = 1; break; }
                out_path = argv[++i];
            } else if (strcmp(argv[i], "--chunk") == 0) {
                if (i + 1 >= argc) { parse_error = 1; break; }
                chunk = atol(argv[++i]);
                if (chunk < 1) { parse_error = 1; break; }
            } else if (strcmp(argv[i], "--scheme") == 0) {
                if (i + 1 >= argc) { parse_error = 1; break; }
                scheme_name = argv[++i];
                if      (!strcmp(scheme_name, "block"))       scheme = SCHEME_BLOCK;
                else if (!strcmp(scheme_name, "cyclic"))      scheme = SCHEME_CYCLIC;
                else if (!strcmp(scheme_name, "blockcyclic")) scheme = SCHEME_BLOCKCYCLIC;
                else if (!strcmp(scheme_name, "dynamic"))     scheme = SCHEME_DYNAMIC;
                else parse_error = 1;
            } else if (argv[i][0] == '-') {
                parse_error = 1;
            } else {
                n = atol(argv[i]);
                have_n = 1;
            }
        }
        if (!have_n || n < 2) parse_error = 1;
        if (parse_error) usage(argv[0]);
    }

    /* Abort cleanly and collectively on a bad command line - if only the root
     * returned, the other ranks would block forever in the broadcast below. */
    MPI_Bcast(&parse_error, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parse_error) {
        MPI_Finalize();
        return 1;
    }

    /* --- Phase: disseminate n (and the run parameters) ------------------- */
    double t0 = MPI_Wtime();

    long params[2] = { n, chunk };
    int  ischeme = (int) scheme;
    MPI_Bcast(params, 2, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ischeme, 1, MPI_INT, 0, MPI_COMM_WORLD);
    n = params[0];
    chunk = params[1];
    scheme = (scheme_t) ischeme;

    double t_bcast = MPI_Wtime() - t0;

    /* A single rank has no one to coordinate with, so master-worker
     * degenerates; fall back to block, which is equivalent at P=1. */
    if (scheme == SCHEME_DYNAMIC && size == 1) {
        scheme = SCHEME_BLOCK;
        scheme_name = "dynamic(->block,P=1)";
    }

    /* --- Phase: allocation ----------------------------------------------- */
    t0 = MPI_Wtime();

    /* Seed each rank's buffer with a bound on what it could possibly find.
     * pi(n) caps every scheme; for the schemes with a known share we take the
     * smaller of that and the rank's candidate count. */
    long global_bound = prime_count_bound(n);
    long seed = global_bound / (size > 0 ? size : 1) + 1024;
    if (seed > global_bound) seed = global_bound;

    vec_t local;
    if (!vec_init(&local, seed)) {
        fprintf(stderr, "rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double t_alloc = MPI_Wtime() - t0;

    /* --- Phase: the distributed prime search ----------------------------- */
    MPI_Barrier(MPI_COMM_WORLD);   /* so t_compute measures the phase, not skew */
    t0 = MPI_Wtime();

    int ok = 1;
    switch (scheme) {

    case SCHEME_BLOCK: {
        /* Contiguous range per rank, remainder spread one-per-rank over the
         * first ranks so block sizes differ by at most 1. */
        long range = n - 2;
        long base = range / size;
        long rem  = range % size;
        long start = 2 + (long) rank * base + (rank < rem ? rank : rem);
        long len   = base + (rank < rem ? 1 : 0);
        ok = scan_range(&local, start, start + len);
        break;
    }

    case SCHEME_CYCLIC: {
        /* Stride by the number of ranks: rank r takes 2+r, 2+r+P, 2+r+2P, ...
         * Costs alternate between ranks, so the sqrt(k) gradient is shared
         * almost perfectly - at the price of poor cache locality. */
        for (long k = 2 + rank; k < n && ok; k += size) {
            if (is_prime(k)) ok = vec_push(&local, k);
        }
        break;
    }

    case SCHEME_BLOCKCYCLIC: {
        /* Chunk c (candidates [2+c*chunk, 2+(c+1)*chunk)) goes to rank
         * c % P. Retains cyclic's balance at chunk granularity while each
         * chunk is walked contiguously. */
        long total = n - 2;
        long nchunks = (total + chunk - 1) / chunk;
        for (long c = rank; c < nchunks && ok; c += size) {
            long start = 2 + c * chunk;
            long end = start + chunk;
            if (end > n) end = n;
            ok = scan_range(&local, start, end);
        }
        break;
    }

    case SCHEME_DYNAMIC: {
        /* Master-worker. Rank 0 owns a cursor over the chunk space and replies
         * to each request with the next chunk index, or -1 to retire the
         * worker. Self-balancing: a rank on a slow core simply asks for fewer
         * chunks, which is what makes this the right scheme on heterogeneous
         * CPUs. Rank 0 does not compute, so P-1 ranks do the work. */
        long total = n - 2;
        long nchunks = (total + chunk - 1) / chunk;

        if (rank == 0) {
            long next = 0;
            int retired = 0;
            while (retired < size - 1) {
                long req;
                MPI_Status st;
                MPI_Recv(&req, 1, MPI_LONG, MPI_ANY_SOURCE, TAG_REQUEST,
                         MPI_COMM_WORLD, &st);
                long reply = (next < nchunks) ? next++ : -1;
                if (reply < 0) retired++;
                MPI_Send(&reply, 1, MPI_LONG, st.MPI_SOURCE, TAG_WORK,
                         MPI_COMM_WORLD);
            }
        } else {
            for (;;) {
                long req = 1, c;
                MPI_Send(&req, 1, MPI_LONG, 0, TAG_REQUEST, MPI_COMM_WORLD);
                MPI_Recv(&c, 1, MPI_LONG, 0, TAG_WORK, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                if (c < 0) break;
                long start = 2 + c * chunk;
                long end = start + chunk;
                if (end > n) end = n;
                if (!scan_range(&local, start, end)) { ok = 0; break; }
            }
        }
        break;
    }
    }

    if (!ok) {
        fprintf(stderr, "rank %d: allocation failed during search\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double t_compute_local = MPI_Wtime() - t0;

    /* MAX is the wall-clock cost of the phase (the slowest rank gates it);
     * MIN lets us report the imbalance rather than merely assert it. */
    double t_compute = 0.0, t_compute_min = 0.0;
    MPI_Reduce(&t_compute_local, &t_compute, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&t_compute_local, &t_compute_min, 1, MPI_DOUBLE, MPI_MIN, 0,
               MPI_COMM_WORLD);

    /* --- Phase: gather results to the root ------------------------------- */
    t0 = MPI_Wtime();

    if (local.count > INT_MAX) {
        fprintf(stderr, "rank %d: local count exceeds MPI_Gatherv int limit\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int local_count = (int) local.count;

    int *counts = NULL, *displs = NULL;
    long *all = NULL;
    long total_count = 0;

    if (rank == 0) {
        counts = malloc(sizeof(int) * (size_t) size);
        displs = malloc(sizeof(int) * (size_t) size);
        if (counts == NULL || displs == NULL) {
            fprintf(stderr, "root: allocation failed for gather metadata\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* Two-step gather: collect the per-rank counts, then the payload. The
     * counts are what let the root size its buffer and build displacements. */
    MPI_Gather(&local_count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long acc = 0;
        for (int i = 0; i < size; i++) {
            displs[i] = (int) acc;
            acc += counts[i];
        }
        total_count = acc;
        all = malloc(sizeof(long) * (size_t) (total_count > 0 ? total_count : 1));
        if (all == NULL) {
            fprintf(stderr, "root: allocation failed for gathered results\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gatherv(local.data, local_count, MPI_LONG,
                all, counts, displs, MPI_LONG, 0, MPI_COMM_WORLD);

    double t_comm = MPI_Wtime() - t0;

    /* --- Phase: ordering -------------------------------------------------- */
    /* Block partitioning assigns ascending contiguous ranges to ascending
     * ranks, so MPI_Gatherv concatenates them already sorted - no work needed.
     * Every other scheme interleaves, so the root must sort. Timed separately
     * so the cost of balance is visible against the cost of imbalance. */
    double t_sort = 0.0;
    if (rank == 0) {
        t0 = MPI_Wtime();
        if (scheme != SCHEME_BLOCK) {
            qsort(all, (size_t) total_count, sizeof(long), cmp_long);
        }
        t_sort = MPI_Wtime() - t0;
    }

    /* --- Phase: file output ---------------------------------------------- */
    double t_io = 0.0;
    if (rank == 0) {
        t0 = MPI_Wtime();
        FILE *fp = fopen(out_path, "w");
        if (fp == NULL) {
            fprintf(stderr, "root: could not open output file '%s'\n", out_path);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (long i = 0; i < total_count; i++) fprintf(fp, "%ld\n", all[i]);
        if (fclose(fp) != 0) {
            fprintf(stderr, "root: failed to flush output file '%s'\n", out_path);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        t_io = MPI_Wtime() - t0;
    }

    double t_total = MPI_Wtime() - t_total_start;

    if (rank == 0) {
        if (csv_mode) {
            /* impl,n,procs,threads,prime_count,
             * t_alloc,t_bcast,t_compute,t_comm,t_sort,t_io,t_total,imbalance */
            printf("mpi-%s,%ld,%d,1,%ld,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                   scheme_name, n, size, total_count,
                   t_alloc, t_bcast, t_compute, t_comm, t_sort, t_io, t_total,
                   t_compute > 0.0 ? t_compute / (t_compute_min > 0.0
                                                  ? t_compute_min : t_compute)
                                   : 1.0);
        } else {
            if (n < 100) {
                printf("Prime numbers less than %ld:\n", n);
                for (long i = 0; i < total_count; i++) printf("%ld ", all[i]);
                printf("\n");
            }
            printf("Found %ld primes less than %ld. Written to %s\n",
                   total_count, n, out_path);
            printf("Processes: %d   scheme: %s   chunk: %ld\n",
                   size, scheme_name, chunk);
            printf("  alloc    : %.6f s\n", t_alloc);
            printf("  bcast    : %.6f s\n", t_bcast);
            printf("  compute  : %.6f s  (slowest rank; fastest %.6f s,"
                   " imbalance %.2fx)\n",
                   t_compute, t_compute_min,
                   t_compute_min > 0.0 ? t_compute / t_compute_min : 1.0);
            printf("  gather   : %.6f s\n", t_comm);
            printf("  sort     : %.6f s%s\n", t_sort,
                   scheme == SCHEME_BLOCK ? "  (not required for block)" : "");
            printf("  file I/O : %.6f s\n", t_io);
            printf("  TOTAL    : %.6f s (Open MPI)\n", t_total);
        }
    }

    free(local.data);
    free(counts);
    free(displs);
    free(all);

    MPI_Finalize();
    return 0;
}
