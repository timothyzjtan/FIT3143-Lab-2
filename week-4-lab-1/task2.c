/*
 * FIT3143 Lab 1 - Task 2: Parallel Prime Number Search with POSIX Threads
 *
 * Utilised Static Block Partitioning to divide the range [2, n) into num_threads blocks.
 * Each thread finds primes in its assigned block and stores them in a private buffer.
 * Ranges of data are assigned to threads based on the number of threads and the total range size, ensuring that any remainder is distributed evenly across the first few threads.
 * Memory is allocated then freed for each thread's private buffer after use.
 * */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

typedef struct {
    long start;   // inclusive
    long end;     // exclusive
    long *primes; // private output buffer for this thread
    long count;
} thread_data_t;

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

void *find_primes_range(void *arg) {
    thread_data_t *data = (thread_data_t *) arg;
    long span = data->end - data->start;    // Calculate the span of the range for this thread
    long capacity = span > 0 ? span : 1;    // To avoid malloc(0) which may lead to unusual behaviour, utilising ternary operators

    data->primes = malloc(sizeof(long) * capacity);
    data->count = 0;

    for (long k = data->start; k < data->end; k++) {
        if (is_prime(k)) {
            data->primes[data->count++] = k;
        }
    }
    return NULL;
}

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

    // Allocates memory for thread handles and their data
    pthread_t *threads = malloc(sizeof(pthread_t) * num_threads);

    // Array of struct that stores the range and output buffer for each thread
    thread_data_t *tdata = malloc(sizeof(thread_data_t) * num_threads);

    /* Static block partitioning of [2, n) across num_threads threads.
     * Any remainder is spread one-per-thread across the first threads
     * so block sizes differ by at most 1. */
    long range_size = n - 2;
    long chunk = range_size / num_threads;
    long remainder = range_size % num_threads;

    long cursor = 2;

    // Points to which section of the range each thread will handle
    for (int t = 0; t < num_threads; t++) {
        long this_chunk = chunk + (t < remainder ? 1 : 0);
        tdata[t].start = cursor;
        tdata[t].end = cursor + this_chunk;
        cursor += this_chunk;
    }

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    // Creates threads to find primes in their assigned ranges
    for (int t = 0; t < num_threads; t++) {
        pthread_create(&threads[t], NULL, find_primes_range, &tdata[t]);
    }

    // Waits for all threads to finish
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = (t_end.tv_sec - t_start.tv_sec) +
                      (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    long total_count = 0;

    // Adds up the primes that were found by the threads
    for (int t = 0; t < num_threads; t++) total_count += tdata[t].count;

    if (n < 100) {
        printf("Prime numbers less than %ld:\n", n);
        for (int t = 0; t < num_threads; t++) {
            for (long i = 0; i < tdata[t].count; i++) {
                printf("%ld ", tdata[t].primes[i]);
            }
        }
        printf("\n");
    } else {
        FILE *fp = fopen("task2_primes_output.txt", "w");
        if (fp == NULL) {
            fprintf(stderr, "Could not open output file.\n");
        } else {
            fprintf(fp, "Prime numbers less than %ld:\n", n);
            for (int t = 0; t < num_threads; t++) {
                for (long i = 0; i < tdata[t].count; i++) {
                    fprintf(fp, "%ld\n", tdata[t].primes[i]);
                }
            }
            fclose(fp);
        }
        printf("Found %ld primes less than %ld. Written to task2_primes_output.txt\n",
               total_count, n);
    }

    printf("Total primes found: %ld\n", total_count);
    printf("Threads used: %d\n", num_threads);
    printf("Time taken (parallel - POSIX Threads): %.6f seconds\n", elapsed);

    // Frees memory
    for (int t = 0; t < num_threads; t++) free(tdata[t].primes);
    free(threads);
    free(tdata);

    return 0;
}