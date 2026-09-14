/*
 * FIT3143 Lab 1 - Task 2: Parallel Prime Number Search with POSIX Threads
 *
 * Utilised Static Block Partitioning to divide the range [2, n) into num_threads blocks.
 * Each thread finds primes in its assigned block and stores them in a private buffer.
 * Ranges of data are assigned to threads based on the number of threads and the total range size,
 * ensuring that any remainder is distributed evenly across the first few threads.
 *
 * Because the sub-ranges are contiguous and increasing, concatenating the
 * per-thread buffers in thread order yields an already-sorted list - no sort
 * is required, so t_merge here is just the concatenation cost.
 *
 * KNOWN LIMITATION (measured, kept deliberately): static block partitioning is
 * load-imbalanced for this problem. Trial division costs ~sqrt(k) iterations,
 * so the highest block does materially more work than the lowest. Measured at
 * n = 10^7 over 8 blocks, the slowest block takes 1.6x the fastest, giving
 * ~80% efficiency and capping speedup at ~6.4x rather than 8x. This scheme is
 * retained as the Week 4 baseline; Lab 2 explores better partitioning.
 *
 * Timing model (see task1.c for the rationale):
 *   t_alloc    per-thread buffer allocation + thread handles   (serial)
 *   t_compute  thread create -> join, i.e. the parallel search (PARALLEL)
 *   t_merge    concatenating per-thread buffers in order       (serial)
 *   t_io       writing the output file                         (serial)
 *   t_total    whole of main()
 *
 * Allocation happens BEFORE the compute timer starts. In the original version
 * each thread malloc'd inside its worker function, so allocation was charged
 * to the parallel phase while the serial baseline paid it outside its timer -
 * which systematically understated the measured speedup.
 *
 * Compile: gcc -O2 -Wall -Wextra task2.c -o task2 -lpthread -lm
 * Run:     ./task2 10000000 4
 *          ./task2 10000000 4 --csv
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

typedef struct {
    long start;   // inclusive
    long end;     // exclusive
    long *primes; // private output buffer for this thread (allocated by main)
    long count;
} thread_data_t;

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + (double) t.tv_nsec / 1e9;
}

/* Upper bound on the number of primes in [start, end).
 *
 * Two independent bounds, whichever is tighter:
 *   - every prime except 2 is odd, so at most span/2 + 2 of them;
 *   - pi(end) < 1.26 * end / ln(end)  (Rosser & Schoenfeld, 1962).
 * The first wins for high blocks, the second for low blocks. */
static long range_prime_bound(long start, long end) {
    long span = end - start;
    if (span <= 0) return 1;

    long odd_bound = span / 2 + 2;
    if (end < 100) return odd_bound;

    long pi_bound = (long) (1.26 * (double) end / log((double) end)) + 16;
    return pi_bound < odd_bound ? pi_bound : odd_bound;
}

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

/* Pure compute - the buffer is already allocated, so this function does no
 * memory management and the compute timer measures only the prime search. */
void *find_primes_range(void *arg) {
    thread_data_t *data = (thread_data_t *) arg;
    data->count = 0;

    for (long k = data->start; k < data->end; k++) {
        if (is_prime(k)) {
            data->primes[data->count++] = k;
        }
    }
    return NULL;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s <n> <num_threads> [--csv] [--out FILE]\n"
        "  <n>            find primes strictly less than n\n"
        "  <num_threads>  number of POSIX threads\n"
        "  --csv          emit one machine-readable CSV row instead of prose\n"
        "  --out FILE     output file (default task2_primes_output.txt)\n",
        prog);
}

int main(int argc, char *argv[]) {
    double t_total_start = now_sec();

    long n = 0;
    int num_threads = 0;
    int csv_mode = 0;
    const char *out_path = "task2_primes_output.txt";

    int positional = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--csv") == 0) {
            csv_mode = 1;
        } else if (strcmp(argv[i], "--out") == 0) {
            if (i + 1 >= argc) { usage(argv[0]); return 1; }
            out_path = argv[++i];
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

    /* --- Phase 1: allocation and partitioning (serial) ------------------- */
    double t0 = now_sec();

    pthread_t *threads = malloc(sizeof(pthread_t) * num_threads);
    thread_data_t *tdata = malloc(sizeof(thread_data_t) * num_threads);
    if (threads == NULL || tdata == NULL) {
        fprintf(stderr, "Memory allocation failed for thread bookkeeping.\n");
        free(threads);
        free(tdata);
        return 1;
    }

    /* Static block partitioning of [2, n) across num_threads threads.
     * Any remainder is spread one-per-thread across the first threads
     * so block sizes differ by at most 1. */
    long range_size = n - 2;
    long chunk = range_size / num_threads;
    long remainder = range_size % num_threads;
    long cursor = 2;

    for (int t = 0; t < num_threads; t++) {
        long this_chunk = chunk + (t < remainder ? 1 : 0);
        tdata[t].start = cursor;
        tdata[t].end = cursor + this_chunk;
        tdata[t].count = 0;
        cursor += this_chunk;

        tdata[t].primes = malloc(sizeof(long) *
                                 range_prime_bound(tdata[t].start, tdata[t].end));
        if (tdata[t].primes == NULL) {
            fprintf(stderr, "Memory allocation failed for thread %d buffer.\n", t);
            for (int u = 0; u < t; u++) free(tdata[u].primes);
            free(threads);
            free(tdata);
            return 1;
        }
    }

    double t_alloc = now_sec() - t0;

    /* --- Phase 2: the parallel prime search ------------------------------ */
    t0 = now_sec();

    for (int t = 0; t < num_threads; t++) {
        int rc = pthread_create(&threads[t], NULL, find_primes_range, &tdata[t]);
        if (rc != 0) {
            fprintf(stderr, "pthread_create failed for thread %d: %s\n",
                    t, strerror(rc));
            /* Join the threads already running so we do not leave them
             * detached and racing against the buffers we are about to free. */
            for (int u = 0; u < t; u++) pthread_join(threads[u], NULL);
            for (int u = 0; u < num_threads; u++) free(tdata[u].primes);
            free(threads);
            free(tdata);
            return 1;
        }
    }

    for (int t = 0; t < num_threads; t++) {
        int rc = pthread_join(threads[t], NULL);
        if (rc != 0) {
            fprintf(stderr, "pthread_join failed for thread %d: %s\n",
                    t, strerror(rc));
            return 1;
        }
    }

    double t_compute = now_sec() - t0;

    /* --- Phase 3: merge (serial) ----------------------------------------- */
    /* Blocks are contiguous and ascending, so concatenating in thread order is
     * already sorted. This copy is the honest serial cost of assembling one
     * result from many workers, and feeds the serial fraction in Amdahl's Law. */
    t0 = now_sec();

    long total_count = 0;
    for (int t = 0; t < num_threads; t++) total_count += tdata[t].count;

    long *all_primes = malloc(sizeof(long) * (total_count > 0 ? total_count : 1));
    if (all_primes == NULL) {
        fprintf(stderr, "Memory allocation failed for merged result.\n");
        for (int t = 0; t < num_threads; t++) free(tdata[t].primes);
        free(threads);
        free(tdata);
        return 1;
    }

    long pos = 0;
    for (int t = 0; t < num_threads; t++) {
        memcpy(all_primes + pos, tdata[t].primes, sizeof(long) * tdata[t].count);
        pos += tdata[t].count;
    }

    double t_merge = now_sec() - t0;

    /* --- Phase 4: file output (serial) ----------------------------------- */
    t0 = now_sec();

    FILE *fp = fopen(out_path, "w");
    if (fp == NULL) {
        fprintf(stderr, "Could not open output file '%s'.\n", out_path);
        free(all_primes);
        for (int t = 0; t < num_threads; t++) free(tdata[t].primes);
        free(threads);
        free(tdata);
        return 1;
    }
    for (long i = 0; i < total_count; i++) {
        fprintf(fp, "%ld\n", all_primes[i]);
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "Failed to flush output file '%s'.\n", out_path);
        free(all_primes);
        for (int t = 0; t < num_threads; t++) free(tdata[t].primes);
        free(threads);
        free(tdata);
        return 1;
    }

    double t_io = now_sec() - t0;
    double t_total = now_sec() - t_total_start;

    if (csv_mode) {
        /* impl,n,procs,threads,prime_count,t_alloc,t_compute,t_merge,t_io,t_total */
        printf("pthreads,%ld,1,%d,%ld,%.6f,%.6f,%.6f,%.6f,%.6f\n",
               n, num_threads, total_count,
               t_alloc, t_compute, t_merge, t_io, t_total);
    } else {
        if (n < 100) {
            printf("Prime numbers less than %ld:\n", n);
            for (long i = 0; i < total_count; i++) printf("%ld ", all_primes[i]);
            printf("\n");
        }
        printf("Found %ld primes less than %ld. Written to %s\n",
               total_count, n, out_path);
        printf("Threads used: %d\n", num_threads);
        printf("  alloc   : %.6f s\n", t_alloc);
        printf("  compute : %.6f s\n", t_compute);
        printf("  merge   : %.6f s\n", t_merge);
        printf("  file I/O: %.6f s\n", t_io);
        printf("  TOTAL   : %.6f s (POSIX Threads)\n", t_total);
    }

    // Frees memory
    free(all_primes);
    for (int t = 0; t < num_threads; t++) free(tdata[t].primes);
    free(threads);
    free(tdata);

    return 0;
}
