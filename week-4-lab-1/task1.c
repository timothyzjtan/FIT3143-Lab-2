/*
 * FIT3143 Lab 1 - Task 1: Serial Prime Number Search
 *
 * Finds all prime numbers strictly less than n (user-supplied) using
 * trial division up to sqrt(k), which avoids the unnecessary checks
 * beyond sqrt(k) (if k = m * d with m <= d, then m <= sqrt(k)).
 *
 * This is the SERIAL BASELINE for every speedup reported in Lab 2. The
 * trial-division algorithm in is_prime() is deliberately frozen: speedup is
 * T_serial / T_parallel, so changing the baseline algorithm (e.g. to a sieve)
 * would silently invalidate every comparison. Only the measurement harness
 * around it has been revised.
 *
 * Timing model - the run is split into four phases so that Lab 2 Task 3 can
 * derive the serial and parallel fractions for Amdahl's / Gustafson's Law:
 *
 *   t_alloc    buffer allocation                      (serial)
 *   t_compute  the prime search itself                (PARALLEL in tasks 2/3)
 *   t_merge    combining per-worker results in order  (serial; zero here)
 *   t_io       writing the output file                (serial)
 *   t_total    whole of main(), including all of the above
 *
 * t_total is the figure used for speedup graphs. It deliberately includes file
 * writing, because the Lab 2 rubric grades "overall speedup (including
 * communication, computation, sorting and file writing)" - a baseline that
 * stopped its clock before the write would not be comparable.
 *
 * Compile: gcc -O2 -Wall -Wextra task1.c -o task1 -lm
 * Run:     ./task1 10000000
 *          ./task1 10000000 --csv
 *          ./task1 10000000 --out primes.txt
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Monotonic wall-clock seconds. CLOCK_MONOTONIC is immune to NTP adjustments,
 * so a clock step mid-benchmark cannot corrupt a measurement. */
static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + (double) t.tv_nsec / 1e9;
}

/* Upper bound on pi(n), the number of primes below n.
 *
 * Rosser & Schoenfeld (1962) give pi(x) < 1.25506 * x / ln(x) for x > 1, so
 * 1.26 is a safe constant; the +16 covers small n where the ratio is coarse.
 * At n = 10^7 this reserves ~782k slots for the 664,579 primes that exist
 * (~6 MB) rather than the ~80 MB that a flat malloc(n) would take. */
static long prime_count_bound(long n) {
    if (n < 100) return 100;
    return (long) (1.26 * (double) n / log((double) n)) + 16;
}

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

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s <n> [--csv] [--out FILE]\n"
        "  <n>         find primes strictly less than n\n"
        "  --csv       emit one machine-readable CSV row instead of prose\n"
        "  --out FILE  output file (default task1_primes_output.txt)\n",
        prog);
}

int main(int argc, char *argv[]) {
    double t_total_start = now_sec();

    long n = 0;
    int csv_mode = 0;
    const char *out_path = "task1_primes_output.txt";

    /* n comes from the command line; the interactive prompt is kept as a
     * fallback so the program still behaves as it did in Week 4. */
    int have_n = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--csv") == 0) {
            csv_mode = 1;
        } else if (strcmp(argv[i], "--out") == 0) {
            if (i + 1 >= argc) { usage(argv[0]); return 1; }
            out_path = argv[++i];
        } else if (argv[i][0] == '-') {
            usage(argv[0]);
            return 1;
        } else {
            n = atol(argv[i]);  // Converts string to long
            have_n = 1;
        }
    }

    if (!have_n) {
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

    /* --- Phase 1: allocation (serial) ------------------------------------ */
    double t0 = now_sec();

    long capacity = prime_count_bound(n);
    long *primes = malloc(sizeof(long) * capacity);
    if (primes == NULL) {
        fprintf(stderr, "Memory allocation failed (%ld longs).\n", capacity);
        return 1;
    }

    double t_alloc = now_sec() - t0;

    /* --- Phase 2: the prime search (the parallelisable part) ------------- */
    long count = 0;
    t0 = now_sec();

    // Loop checks all numbers from 2 to n-1
    for (long k = 2; k < n; k++) {
        if (is_prime(k)) {
            primes[count++] = k;
        }
    }

    double t_compute = now_sec() - t0;

    /* --- Phase 3: merge (serial) ----------------------------------------- */
    /* The serial search emits primes in ascending order already, so there is
     * nothing to merge. Measured anyway, and reported as zero, so the phase
     * breakdown lines up column-for-column with tasks 2 and 3. */
    double t_merge = 0.0;

    /* --- Phase 4: file output (serial) ----------------------------------- */
    t0 = now_sec();

    FILE *fp = fopen(out_path, "w");
    if (fp == NULL) {
        fprintf(stderr, "Could not open output file '%s'.\n", out_path);
        free(primes);
        return 1;
    }
    for (long i = 0; i < count; i++) {
        fprintf(fp, "%ld\n", primes[i]);
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "Failed to flush output file '%s'.\n", out_path);
        free(primes);
        return 1;
    }

    double t_io = now_sec() - t0;

    double t_total = now_sec() - t_total_start;

    if (csv_mode) {
        /* impl,n,procs,threads,prime_count,t_alloc,t_compute,t_merge,t_io,t_total */
        printf("serial,%ld,1,1,%ld,%.6f,%.6f,%.6f,%.6f,%.6f\n",
               n, count, t_alloc, t_compute, t_merge, t_io, t_total);
    } else {
        // Small inputs are echoed to the terminal as well as the file
        if (n < 100) {
            printf("Prime numbers less than %ld:\n", n);
            for (long i = 0; i < count; i++) printf("%ld ", primes[i]);
            printf("\n");
        }
        printf("Found %ld primes less than %ld. Written to %s\n",
               count, n, out_path);
        printf("  alloc   : %.6f s\n", t_alloc);
        printf("  compute : %.6f s\n", t_compute);
        printf("  merge   : %.6f s\n", t_merge);
        printf("  file I/O: %.6f s\n", t_io);
        printf("  TOTAL   : %.6f s (serial)\n", t_total);
    }

    free(primes);
    return 0;
}
