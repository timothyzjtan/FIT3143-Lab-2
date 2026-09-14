/*
 * FIT3143 Lab 1 - Task 3: Parallel Prime Number Search with OpenMP
 *
 * Parallel partitioning scheme:
 *   Same static block partitioning as Task 2, but expressed with an
 *   OpenMP parallel region (`#pragma omp parallel`) instead of manual
 *   pthread management. Each thread computes its own contiguous
 *   sub-range of [2, n) from its thread id (omp_get_thread_num()) and
 *   the total thread count (omp_get_num_threads()), then writes its
 *   results into a private buffer - mirroring Task 2 so the two
 *   implementations are directly comparable.
 *
 *   Because the sub-ranges are contiguous and increasing, concatenating
 *   the per-thread buffers in thread order gives an already sorted list.
 *
 * Compile: gcc task3.c -o task3 -fopenmp -lm
 * Run:     ./task3                    (interactive prompts)
 *          ./task3 10000000 4         (n and thread count via CLI args)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* Trial division up to sqrt(k): if k = m * d with m <= d, then m can never
 * exceed sqrt(k), so any factor beyond that would already have been caught
 * by its smaller pair - checking further is redundant. */
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

/* Reads n and the thread count (CLI args or interactive prompt), then
 * finds all primes in [2, n) using OpenMP: each thread searches its own
 * static contiguous sub-range and appends results into a private buffer,
 * so no locks are needed and concatenating in thread order yields an
 * already-sorted list. Prints to stdout for n < 100, otherwise writes to
 * task3_primes_output.txt. */
int main(int argc, char *argv[]) {
    long n;
    int num_threads;

    if (argc == 3) {
        n = atol(argv[1]);
        num_threads = atoi(argv[2]);
    } else {
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

    long **thread_primes = malloc(sizeof(long *) * num_threads);
    long *thread_counts = malloc(sizeof(long) * num_threads);
    if (thread_primes == NULL || thread_counts == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }

    double t_start = omp_get_wtime();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        /* Static block partitioning of [2, n), remainder spread over
         * the first `remainder` threads so block sizes differ by at
         * most 1 - identical scheme to Task 2. */
        long range_size = n - 2;
        long chunk = range_size / nthreads;
        long remainder = range_size % nthreads;

        long my_start = 2 + tid * chunk + (tid < remainder ? tid : remainder);
        long my_chunk = chunk + (tid < remainder ? 1 : 0);
        long my_end = my_start + my_chunk;

        long capacity = my_chunk > 0 ? my_chunk : 1;
        long *local_primes = malloc(sizeof(long) * capacity);
        if (local_primes == NULL) {
            fprintf(stderr, "Memory allocation failed in thread %d.\n", tid);
            exit(1);
        }
        long local_count = 0;

        for (long k = my_start; k < my_end; k++) {
            if (is_prime(k)) {
                local_primes[local_count++] = k;
            }
        }

        thread_primes[tid] = local_primes;
        thread_counts[tid] = local_count;
    }

    double t_end = omp_get_wtime();
    double elapsed = t_end - t_start;

    long total_count = 0;
    for (int t = 0; t < num_threads; t++) total_count += thread_counts[t];

    if (n < 100) {
        printf("Prime numbers less than %ld:\n", n);
        for (int t = 0; t < num_threads; t++) {
            for (long i = 0; i < thread_counts[t]; i++) {
                printf("%ld ", thread_primes[t][i]);
            }
        }
        printf("\n");
    } else {
        FILE *fp = fopen("task3_primes_output.txt", "w");
        if (fp == NULL) {
            fprintf(stderr, "Could not open output file.\n");
        } else {
            fprintf(fp, "Prime numbers less than %ld:\n", n);
            for (int t = 0; t < num_threads; t++) {
                for (long i = 0; i < thread_counts[t]; i++) {
                    fprintf(fp, "%ld\n", thread_primes[t][i]);
                }
            }
            fclose(fp);
        }
        printf("Found %ld primes less than %ld. Written to task3_primes_output.txt\n",
               total_count, n);
    }

    printf("Total primes found: %ld\n", total_count);
    printf("Threads used: %d\n", num_threads);
    printf("Time taken (parallel - OpenMP): %.6f seconds\n", elapsed);

    for (int t = 0; t < num_threads; t++) free(thread_primes[t]);
    free(thread_primes);
    free(thread_counts);

    return 0;
}
