/*
 * FIT3143 Lab 1 - Task 1: Serial Prime Number Search
 *
 * Finds all prime numbers strictly less than n (user-supplied) using
 * trial division up to sqrt(k), which avoids the unnecessary checks
 * beyond sqrt(k) (if k = m * d with m <= d, then m <= sqrt(k)).
 *
 * Output:
 *   n < 100  -> printed to stdout
 *   n >= 100 -> written to task1_primes_output.txt
 * */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* Edge cases for prime checking. */
int is_prime(long k) {
    if (k < 2) return 0;
    if (k == 2) return 1;
    if (k % 2 == 0) return 0;

    // Checks for factors with square root to reduce the number of iterations
    long limit = (long) sqrt((double) k);
    for (long i = 3; i <= limit; i += 2) {
        if (k % i == 0) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    long n;

    // Allows user to input n via command line or interactive prompt
    if (argc == 2) {
        n = atol(argv[1]);  // Converts string to long
    } else {
        printf("Enter n (find primes strictly less than n): ");
        if (scanf("%ld", &n) != 1) {
            fprintf(stderr, "Invalid input.\n");
            return 1;
        }
    }

    if (n < 2) {
        printf("There are no prime numbers strictly less than %ld.\n", n);
        return 0;
    }

    long *primes = malloc(sizeof(long) * n);
    if (primes == NULL) {
        fprintf(stderr, "Memory allocation failed.\n"); // Throws an error msg
        return 1;
    }

    long count = 0;

    struct timespec t_start, t_end; // Stores a time value with seconds and nanoseconds
    clock_gettime(CLOCK_MONOTONIC, &t_start);   // Records time at the start

    // Loop checks all numbers from 2 to n-1
    for (long k = 2; k < n; k++) {
        if (is_prime(k)) {
            primes[count++] = k;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end); // Records time at the end

    // Computes both seconds and nanoseconds
    double elapsed = (t_end.tv_sec - t_start.tv_sec) +
                      (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    // Directly prints the primes 
    if (n < 100) {
        printf("Prime numbers less than %ld:\n", n);
        for (long i = 0; i < count; i++) {
            printf("%ld ", primes[i]);
        }
        printf("\n");
    } else {
        // Makes a text file for n > 100
        FILE *fp = fopen("task1_primes_output.txt", "w");
        if (fp == NULL) {
            fprintf(stderr, "Could not open output file.\n");
            free(primes);
            return 1;
        }
        fprintf(fp, "Prime numbers less than %ld:\n", n);
        for (long i = 0; i < count; i++) {
            fprintf(fp, "%ld\n", primes[i]);
        }
        fclose(fp);
        printf("Found %ld primes less than %ld. Written to task1_primes_output.txt\n",
               count, n);
    }

    printf("Total primes found: %ld\n", count);
    printf("Time taken (serial): %.6f seconds\n", elapsed);

    free(primes);
    return 0;
}