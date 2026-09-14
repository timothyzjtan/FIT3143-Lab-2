/*
 * FIT3143 Lab 2 - Task 2: Prime Search with hybrid Open MPI + OpenMP
 *
 * Two levels of parallelism:
 *   - DISTRIBUTED memory across MPI processes (ranks), which do not share an
 *     address space and must exchange results through messages;
 *   - SHARED memory within each rank, where OpenMP threads cooperate over that
 *     rank's share of the candidates without any copying.
 *
 * The trial-division kernel is identical to every other implementation in
 * Lab 1 and Lab 2, so speedup measures parallelism rather than algorithm.
 *
 * ---------------------------------------------------------------------------
 * WORK DECOMPOSITION
 *
 * The candidate range [2, n) is cut into fixed-size chunks of --chunk
 * candidates. A chunk is the unit of work at BOTH levels, which keeps the two
 * levels from fighting each other:
 *
 *   MPI level     decides which chunks a rank owns (--scheme)
 *   OpenMP level  distributes that rank's chunks across its threads with
 *                 `#pragma omp for schedule(dynamic)`
 *
 * Schemes (--scheme):
 *   block        Rank r owns one contiguous run of chunks. Gathers in rank
 *                order already sorted, so no global sort is needed.
 *   blockcyclic  Chunk c goes to rank c % P. Interleaving shares the sqrt(k)
 *                cost gradient; needs a global sort.
 *   dynamic      Master-worker. Rank 0 hands out BATCHES of chunks on request,
 *                one batch being (threads x chunk) candidates so that every
 *                worker still has enough work to occupy all of its threads.
 *                Adapts to cores of differing speed - which matters here,
 *                since the host has 4 performance + 6 efficiency cores.
 *
 * The pure-cyclic scheme from Task 1 is deliberately absent: it strides per
 * candidate rather than per chunk, so it has no natural chunk decomposition
 * for the OpenMP level to distribute.
 *
 * ---------------------------------------------------------------------------
 * RESULT ORDERING
 *
 * Within a rank, threads pull chunks dynamically, so their private buffers
 * hold scattered values. Each rank therefore sorts its own results before
 * gathering. That local sort runs CONCURRENTLY on every rank, so it is real
 * parallel work rather than an added serial cost.
 *
 * After the gather, a global sort at the root is needed only when the scheme
 * interleaves chunks between ranks. Under `block`, the ranks' already-sorted
 * runs concatenate in ascending order, so the root sorts nothing at all.
 *
 * ---------------------------------------------------------------------------
 * TIMING MODEL (MPI_Wtime; t_compute and t_lsort reduced with MPI_MAX because
 * the slowest rank gates a parallel phase)
 *
 *   t_alloc    per-thread and per-rank buffers            serial
 *   t_bcast    disseminating n to all ranks               serial
 *   t_compute  the threaded, distributed search           PARALLEL
 *   t_lsort    per-rank sort of local results             PARALLEL (across ranks)
 *   t_comm     MPI_Gather + MPI_Gatherv                   serial
 *   t_sort     global sort at the root (0 for block)      serial
 *   t_io       writing the output file                    serial
 *   t_total    whole run
 *
 * Compile: mpicc -O2 -Wall -Wextra task2.c -o task2 -fopenmp -lm
 * Run:     mpirun -np 4 ./task2 10000000 --threads 2
 *          mpirun -np 4 ./task2 10000000 --threads 2 --scheme dynamic --csv
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <mpi.h>
#include <omp.h>

#define TAG_REQUEST 1
#define TAG_WORK    2

typedef enum { SCHEME_BLOCK, SCHEME_BLOCKCYCLIC, SCHEME_DYNAMIC } scheme_t;

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

/* pi(x) < 1.25506 x / ln(x) for x > 1 (Rosser & Schoenfeld, 1962). */
static long prime_count_bound(long n) {
    if (n < 100) return 100;
    return (long) (1.26 * (double) n / log((double) n)) + 16;
}

/* Growable buffer. Each THREAD owns one, so appends never race and no lock is
 * needed on the hot path. Doubling keeps appends amortised O(1). */
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

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: mpirun -np P %s <n> [options]\n"
        "  <n>             find primes strictly less than n\n"
        "  --threads T     OpenMP threads per MPI process (default 2)\n"
        "  --scheme KIND   block | blockcyclic | dynamic (default dynamic)\n"
        "  --chunk N       candidates per chunk (default 4096)\n"
        "  --csv           emit one machine-readable CSV row\n"
        "  --out FILE      output file (default task2_hybrid_output.txt)\n",
        prog);
}

int main(int argc, char *argv[]) {
    /* Ask for thread support: the master-worker scheme calls MPI from inside a
     * rank that also runs OpenMP regions. All MPI calls here are made from the
     * master thread only, so MPI_THREAD_FUNNELED is the level required. */
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    double t_total_start = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (provided < MPI_THREAD_FUNNELED && rank == 0) {
        fprintf(stderr, "warning: MPI provides thread level %d, wanted %d\n",
                provided, MPI_THREAD_FUNNELED);
    }

    long n = 0, chunk = 4096;
    int nthreads = 2, csv_mode = 0;
    scheme_t scheme = SCHEME_DYNAMIC;
    const char *scheme_name = "dynamic";
    const char *out_path = "task2_hybrid_output.txt";

    int parse_error = 0;
    if (rank == 0) {
        int have_n = 0;
        for (int i = 1; i < argc && !parse_error; i++) {
            if (!strcmp(argv[i], "--csv")) {
                csv_mode = 1;
            } else if (!strcmp(argv[i], "--out")) {
                if (i + 1 >= argc) { parse_error = 1; break; }
                out_path = argv[++i];
            } else if (!strcmp(argv[i], "--chunk")) {
                if (i + 1 >= argc) { parse_error = 1; break; }
                chunk = atol(argv[++i]);
                if (chunk < 1) parse_error = 1;
            } else if (!strcmp(argv[i], "--threads")) {
                if (i + 1 >= argc) { parse_error = 1; break; }
                nthreads = atoi(argv[++i]);
                if (nthreads < 1) parse_error = 1;
            } else if (!strcmp(argv[i], "--scheme")) {
                if (i + 1 >= argc) { parse_error = 1; break; }
                scheme_name = argv[++i];
                if      (!strcmp(scheme_name, "block"))       scheme = SCHEME_BLOCK;
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

    /* Collective abort on a bad command line: a lone early return on the root
     * would leave every other rank blocked in the broadcast below. */
    MPI_Bcast(&parse_error, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parse_error) { MPI_Finalize(); return 1; }

    /* --- Phase: disseminate the run parameters --------------------------- */
    double t0 = MPI_Wtime();

    long params[2] = { n, chunk };
    int  meta[2] = { (int) scheme, nthreads };
    MPI_Bcast(params, 2, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(meta, 2, MPI_INT, 0, MPI_COMM_WORLD);
    n = params[0]; chunk = params[1];
    scheme = (scheme_t) meta[0]; nthreads = meta[1];

    double t_bcast = MPI_Wtime() - t0;

    /* Master-worker needs someone to coordinate; at P=1 it degenerates. */
    if (scheme == SCHEME_DYNAMIC && size == 1) {
        scheme = SCHEME_BLOCK;
        scheme_name = "dynamic(->block,P=1)";
    }

    omp_set_num_threads(nthreads);

    /* --- Phase: allocation ----------------------------------------------- */
    t0 = MPI_Wtime();

    long global_bound = prime_count_bound(n);
    long seed = global_bound / ((long) size * nthreads) + 1024;
    if (seed > global_bound) seed = global_bound;

    vec_t *tvec = malloc(sizeof(vec_t) * (size_t) nthreads);
    if (tvec == NULL) {
        fprintf(stderr, "rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int t = 0; t < nthreads; t++) {
        if (!vec_init(&tvec[t], seed)) {
            fprintf(stderr, "rank %d: thread buffer allocation failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    double t_alloc = MPI_Wtime() - t0;

    long total_chunks = ((n - 2) + chunk - 1) / chunk;

    /* --- Phase: the threaded, distributed search -------------------------- */
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();

    int alloc_fail = 0;

    if (scheme == SCHEME_DYNAMIC && rank == 0) {
        /* Master: hand out batches of chunks, then retire each worker. A batch
         * is `nthreads` chunks so a worker always receives enough work to fill
         * all of its threads; handing out single chunks would leave most
         * threads idle while the rank round-tripped to the master. */
        long next = 0;
        int retired = 0;
        while (retired < size - 1) {
            long req;
            MPI_Status st;
            MPI_Recv(&req, 1, MPI_LONG, MPI_ANY_SOURCE, TAG_REQUEST,
                     MPI_COMM_WORLD, &st);
            long reply = (next < total_chunks) ? next : -1;
            if (reply < 0) retired++; else next += nthreads;
            MPI_Send(&reply, 1, MPI_LONG, st.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD);
        }
    } else if (scheme == SCHEME_DYNAMIC) {
        /* Worker: request a batch, spread its chunks across threads, repeat. */
        for (;;) {
            long req = 1, base;
            MPI_Send(&req, 1, MPI_LONG, 0, TAG_REQUEST, MPI_COMM_WORLD);
            MPI_Recv(&base, 1, MPI_LONG, 0, TAG_WORK, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            if (base < 0) break;

            long last = base + nthreads;
            if (last > total_chunks) last = total_chunks;

            #pragma omp parallel for schedule(dynamic, 1)
            for (long c = base; c < last; c++) {
                vec_t *v = &tvec[omp_get_thread_num()];
                long s = 2 + c * chunk, e = s + chunk;
                if (e > n) e = n;
                for (long k = s; k < e; k++) {
                    if (is_prime(k) && !vec_push(v, k)) {
                        #pragma omp atomic write
                        alloc_fail = 1;
                    }
                }
            }
        }
    } else {
        /* Static schemes: the rank's chunk set is known up front, so a single
         * OpenMP loop covers it. schedule(dynamic) lets threads that draw
         * cheap chunks come back for more, which matters both because the
         * sqrt(k) cost rises with k and because the cores are not equal. */
        #pragma omp parallel for schedule(dynamic, 1)
        for (long c = 0; c < total_chunks; c++) {
            int mine;
            if (scheme == SCHEME_BLOCK) {
                long per = (total_chunks + size - 1) / size;
                mine = (c / per == (long) rank);
            } else {
                mine = (c % (long) size == (long) rank);
            }
            if (!mine) continue;

            vec_t *v = &tvec[omp_get_thread_num()];
            long s = 2 + c * chunk, e = s + chunk;
            if (e > n) e = n;
            for (long k = s; k < e; k++) {
                if (is_prime(k) && !vec_push(v, k)) {
                    #pragma omp atomic write
                    alloc_fail = 1;
                }
            }
        }
    }

    if (alloc_fail) {
        fprintf(stderr, "rank %d: allocation failed during search\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double t_compute_local = MPI_Wtime() - t0;

    /* --- Phase: per-rank concatenate + sort (parallel across ranks) ------- */
    t0 = MPI_Wtime();

    long local_total = 0;
    for (int t = 0; t < nthreads; t++) local_total += tvec[t].count;

    long *local = malloc(sizeof(long) * (size_t) (local_total > 0 ? local_total : 1));
    if (local == NULL) {
        fprintf(stderr, "rank %d: allocation failed for local merge\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    long pos = 0;
    for (int t = 0; t < nthreads; t++) {
        memcpy(local + pos, tvec[t].data, sizeof(long) * (size_t) tvec[t].count);
        pos += tvec[t].count;
    }
    /* Threads pulled chunks dynamically, so this rank's values are unordered.
     * Sorting here rather than at the root keeps the work distributed. */
    qsort(local, (size_t) local_total, sizeof(long), cmp_long);

    double t_lsort_local = MPI_Wtime() - t0;

    double t_compute = 0.0, t_compute_min = 0.0, t_lsort = 0.0;
    MPI_Reduce(&t_compute_local, &t_compute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_compute_local, &t_compute_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_lsort_local, &t_lsort, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* --- Phase: gather to the root --------------------------------------- */
    t0 = MPI_Wtime();

    if (local_total > INT_MAX) {
        fprintf(stderr, "rank %d: local count exceeds MPI_Gatherv int limit\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int local_count = (int) local_total;

    int *counts = NULL, *displs = NULL;
    long *all = NULL, total_count = 0;

    if (rank == 0) {
        counts = malloc(sizeof(int) * (size_t) size);
        displs = malloc(sizeof(int) * (size_t) size);
        if (counts == NULL || displs == NULL) {
            fprintf(stderr, "root: allocation failed for gather metadata\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gather(&local_count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long acc = 0;
        for (int i = 0; i < size; i++) { displs[i] = (int) acc; acc += counts[i]; }
        total_count = acc;
        all = malloc(sizeof(long) * (size_t) (total_count > 0 ? total_count : 1));
        if (all == NULL) {
            fprintf(stderr, "root: allocation failed for gathered results\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gatherv(local, local_count, MPI_LONG,
                all, counts, displs, MPI_LONG, 0, MPI_COMM_WORLD);

    double t_comm = MPI_Wtime() - t0;

    /* --- Phase: global ordering ------------------------------------------ */
    /* Under `block`, rank r owns a contiguous, ascending run of chunks and has
     * already sorted it, so concatenation in rank order is globally sorted and
     * nothing remains to do. The interleaving schemes need one final sort. */
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
            fprintf(stderr, "root: failed to flush '%s'\n", out_path);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        t_io = MPI_Wtime() - t0;
    }

    double t_total = MPI_Wtime() - t_total_start;

    if (rank == 0) {
        if (csv_mode) {
            /* impl,n,procs,threads,prime_count,t_alloc,t_bcast,t_compute,
             * t_lsort,t_comm,t_sort,t_io,t_total,imbalance,total_workers */
            printf("hybrid-%s,%ld,%d,%d,%ld,"
                   "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n",
                   scheme_name, n, size, nthreads, total_count,
                   t_alloc, t_bcast, t_compute, t_lsort, t_comm, t_sort, t_io,
                   t_total,
                   t_compute_min > 0.0 ? t_compute / t_compute_min : 1.0,
                   size * nthreads);
        } else {
            if (n < 100) {
                printf("Prime numbers less than %ld:\n", n);
                for (long i = 0; i < total_count; i++) printf("%ld ", all[i]);
                printf("\n");
            }
            printf("Found %ld primes less than %ld. Written to %s\n",
                   total_count, n, out_path);
            printf("Processes: %d  threads/process: %d  total workers: %d\n",
                   size, nthreads, size * nthreads);
            printf("Scheme: %s  chunk: %ld\n", scheme_name, chunk);
            printf("  alloc     : %.6f s\n", t_alloc);
            printf("  bcast     : %.6f s\n", t_bcast);
            printf("  compute   : %.6f s  (slowest rank; fastest %.6f s,"
                   " imbalance %.2fx)\n", t_compute, t_compute_min,
                   t_compute_min > 0.0 ? t_compute / t_compute_min : 1.0);
            printf("  local sort: %.6f s  (concurrent across ranks)\n", t_lsort);
            printf("  gather    : %.6f s\n", t_comm);
            printf("  glob sort : %.6f s%s\n", t_sort,
                   scheme == SCHEME_BLOCK ? "  (not required for block)" : "");
            printf("  file I/O  : %.6f s\n", t_io);
            printf("  TOTAL     : %.6f s (hybrid MPI+OpenMP)\n", t_total);
        }
    }

    for (int t = 0; t < nthreads; t++) free(tvec[t].data);
    free(tvec);
    free(local);
    free(counts);
    free(displs);
    free(all);

    MPI_Finalize();
    return 0;
}
