/*
 * FIT3143 Lab 1 - Task 3: Parallel Prime Number Search with OpenMP
 *
 * Parallel partitioning scheme:
 *   Work is distributed by an OpenMP WORK-SHARING construct
 *   (`#pragma omp parallel for schedule(runtime)`), not by hand. The schedule
 *   kind and chunk size are set at run time via omp_set_schedule(), so
 *   static / dynamic / guided can be compared without recompiling.
 *
 *   This replaces the earlier version, which computed each thread's
 *   sub-range manually from omp_get_thread_num() inside a bare
 *   `#pragma omp parallel`. That approach re-implemented the pthread
 *   partitioning by hand and never used OpenMP's scheduler at all, so it
 *   inherited the same load imbalance: trial division costs ~sqrt(k)
 *   iterations, and with static blocks the highest block does ~1.6x the work
 *   of the lowest (measured at n = 10^7, 8 threads), capping speedup at
 *   ~6.4x instead of 8x. A dynamic schedule lets idle threads steal later
 *   chunks and closes most of that gap.
 *
 * Result ordering:
 *   With a dynamic schedule a thread receives scattered, non-contiguous
 *   chunks, so per-thread buffers can no longer simply be concatenated in
 *   thread order. Instead each iteration writes into a shared flag array at
 *   its own index k - distinct indices, therefore no data race and no locks -
 *   and a serial sweep afterwards compacts the flags into ascending order.
 *
 *   That compaction is an ADDED SERIAL PHASE. It is measured as t_merge and
 *   reported separately, which is exactly the serial fraction Amdahl's Law
 *   needs; Lab 2 Task 3 requires designing an experiment to measure it.
 *
 * Timing model (see task1.c for the rationale):
 *   t_alloc    flag array + result buffer allocation   (serial)
 *   t_compute  the parallel for over [2, n)            (PARALLEL)
 *   t_merge    serial compaction of flags into order   (serial)
 *   t_io       writing the output file                 (serial)
 *   t_total    whole of main()
 *
 * Compile: gcc -O2 -Wall -Wextra task3.c -o task3 -fopenmp -lm
 * Run:     ./task3 10000000 4
 *          ./task3 10000000 4 --schedule dynamic --chunk 4096
 *          ./task3 10000000 4 --csv
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* Upper bound on pi(n): pi(x) < 1.25506 * x / ln(x) for x > 1
 * (Rosser & Schoenfeld, 1962). At n = 10^7 this reserves ~782k slots for the
 * 664,579 primes that exist, rather than the n slots a flat malloc would take. */
static long prime_count_bound(long n) {
    if (n < 100) return 100;
    return (long) (1.26 * (double) n / log((double) n)) + 16;
}

/* Trial division up to sqrt(k): if k = m * d with m <= d, then m can never
 * exceed sqrt(k), so any factor beyond that would already have been caught
 * by its smaller pair - checking further is redundant.
 *
 * Identical to task1.c and task2.c by design: speedup is T_serial/T_parallel,
 * so the per-candidate work must be the same in every implementation for the
 * comparison to isolate parallelism rather than algorithm. */
int is_prime(long k) {
    if (k < 2) return 0;
    if (k == 2) return 1;
    if (k % 2 == 0) return 0;

    long limit = (long) sqrt((double) k);
    for (long i = 3; i <= limit; i += 2) {
        if (k % i == 0) return 0;
    }
    return 1;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s <n> <num_threads> [--schedule KIND] [--chunk N] [--csv] [--out FILE]\n"
        "  <n>              find primes strictly less than n\n"
        "  <num_threads>    number of OpenMP threads\n"
        "  --schedule KIND  static | dynamic | guided | auto  (default dynamic)\n"
        "  --chunk N        chunk size for the schedule       (default 4096)\n"
        "  --csv            emit one machine-readable CSV row instead of prose\n"
        "  --out FILE       output file (default task3_primes_output.txt)\n",
        prog);
}

int main(int argc, char *argv[]) {
    double t_total_start = omp_get_wtime();

    long n = 0;
    int num_threads = 0;
    int csv_mode = 0;
    const char *out_path = "task3_primes_output.txt";
    const char *sched_name = "dynamic";
    omp_sched_t sched_kind = omp_sched_dynamic;
    int chunk_size = 4096;

    int positional = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--csv") == 0) {
            csv_mode = 1;
        } else if (strcmp(argv[i], "--out") == 0) {
            if (i + 1 >= argc) { usage(argv[0]); return 1; }
            out_path = argv[++i];
        } else if (strcmp(argv[i], "--chunk") == 0) {
            if (i + 1 >= argc) { usage(argv[0]); return 1; }
            chunk_size = atoi(argv[++i]);
            if (chunk_size < 0) { fprintf(stderr, "--chunk must be >= 0\n"); return 1; }
            /* chunk 0 means "the implementation's default for this kind".
             * It matters: schedule(static) with the default chunk gives each
             * thread ONE contiguous block, which is the load-imbalanced scheme
             * we want to be able to measure. Any explicit chunk instead deals
             * chunks round-robin, which balances the sqrt(k) cost gradient
             * almost as well as a dynamic schedule does. */
        } else if (strcmp(argv[i], "--schedule") == 0) {
            if (i + 1 >= argc) { usage(argv[0]); return 1; }
            sched_name = argv[++i];
            if      (strcmp(sched_name, "static")  == 0) sched_kind = omp_sched_static;
            else if (strcmp(sched_name, "dynamic") == 0) sched_kind = omp_sched_dynamic;
            else if (strcmp(sched_name, "guided")  == 0) sched_kind = omp_sched_guided;
            else if (strcmp(sched_name, "auto")    == 0) sched_kind = omp_sched_auto;
            else { fprintf(stderr, "Unknown schedule '%s'\n", sched_name); return 1; }
        } else if (argv[i][0] == '-') {
            usage(argv[0]);
            return 1;
        } else if (positional == 0) {
            n = atol(argv[i]);
            positional++;
        } else if (positional == 1) {
            num_threads = atoi(argv[i]);
            positional++;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (positional < 2) {
        printf("Enter n (find primes strictly less than n): ");
        if (scanf("%ld", &n) != 1) { fprintf(stderr, "Invalid input.\n"); return 1; }
        printf("Enter number of threads: ");
        if (scanf("%d", &num_threads) != 1) { fprintf(stderr, "Invalid input.\n"); return 1; }
    }

    if (n < 2 || num_threads < 1) {
        printf("Invalid input (need n >= 2 and at least 1 thread).\n");
        return 0;
    }

    omp_set_num_threads(num_threads);
    /* schedule(runtime) below defers to this, so the partitioning strategy is
     * a run-time experiment variable rather than a compile-time constant. */
    omp_set_schedule(sched_kind, chunk_size);

    /* --- Phase 1: allocation (serial) ------------------------------------ */
    double t0 = omp_get_wtime();

    /* One byte per candidate. Indexed directly by k so that each loop
     * iteration touches a distinct element - that is what makes the parallel
     * loop race-free without any synchronisation. */
    char *flags = calloc((size_t) n, sizeof(char));
    long capacity = prime_count_bound(n);
    long *primes = malloc(sizeof(long) * capacity);
    if (flags == NULL || primes == NULL) {
        fprintf(stderr, "Memory allocation failed (n = %ld).\n", n);
        free(flags);
        free(primes);
        return 1;
    }

    double t_alloc = omp_get_wtime() - t0;

    /* --- Phase 2: the parallel prime search ------------------------------ */
    t0 = omp_get_wtime();

    #pragma omp parallel for schedule(runtime)
    for (long k = 2; k < n; k++) {
        flags[k] = (char) is_prime(k);
    }

    double t_compute = omp_get_wtime() - t0;

    /* --- Phase 3: merge / compaction (serial) ---------------------------- */
    /* Sweeping the flag array in index order produces ascending primes by
     * construction, so no sort is needed regardless of which thread found
     * which candidate. This is the serial phase referred to in the header. */
    t0 = omp_get_wtime();

    long count = 0;
    for (long k = 2; k < n; k++) {
        if (flags[k]) primes[count++] = k;
    }

    double t_merge = omp_get_wtime() - t0;

    /* --- Phase 4: file output (serial) ----------------------------------- */
    t0 = omp_get_wtime();

    FILE *fp = fopen(out_path, "w");
    if (fp == NULL) {
        fprintf(stderr, "Could not open output file '%s'.\n", out_path);
        free(flags);
        free(primes);
        return 1;
    }
    for (long i = 0; i < count; i++) {
        fprintf(fp, "%ld\n", primes[i]);
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "Failed to flush output file '%s'.\n", out_path);
        free(flags);
        free(primes);
        return 1;
    }

    double t_io = omp_get_wtime() - t0;
    double t_total = omp_get_wtime() - t_total_start;

    if (csv_mode) {
        /* impl,n,procs,threads,prime_count,t_alloc,t_compute,t_merge,t_io,t_total
         * The schedule is folded into the impl field so a sweep across
         * schedules stays distinguishable in one results file. */
        printf("openmp-%s-%d,%ld,1,%d,%ld,%.6f,%.6f,%.6f,%.6f,%.6f\n",
               sched_name, chunk_size, n, num_threads, count,
               t_alloc, t_compute, t_merge, t_io, t_total);
    } else {
        if (n < 100) {
            printf("Prime numbers less than %ld:\n", n);
            for (long i = 0; i < count; i++) printf("%ld ", primes[i]);
            printf("\n");
        }
        printf("Found %ld primes less than %ld. Written to %s\n",
               count, n, out_path);
        printf("Threads used: %d  schedule: %s,%d\n",
               num_threads, sched_name, chunk_size);
        printf("  alloc   : %.6f s\n", t_alloc);
        printf("  compute : %.6f s\n", t_compute);
        printf("  merge   : %.6f s\n", t_merge);
        printf("  file I/O: %.6f s\n", t_io);
        printf("  TOTAL   : %.6f s (OpenMP)\n", t_total);
    }

    free(flags);
    free(primes);
    return 0;
}
