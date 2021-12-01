// Uses OpenMP for parallelization and enables vectorization through use
// of GCC ivdep pragma

// Based on the approach in Ulrich Drepper's What Every Programmer Should
// Know About Memory:
// https://people.freebsd.org/~lstewart/articles/cpumemory.pdf

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <emmintrin.h>

// Print message describing the error and die
#define handle_error(msg)                                                      \
  do {                                                                         \
    fprintf(stderr, "%s: %s (%s:%d)\n", (msg), strerror(errno), __FILE__,      \
            __LINE__);                                                         \
    exit(EXIT_FAILURE);                                                        \
  } while (0)

// Convenience macro for indexing matrix
#define IDX(N, r, c) ((N) * (r) + (c))

// Cache line size
long g_linesize;

// Used for tic() and toc()
struct timeval g_t1, g_t2;

// Simple timer function. tic() records the current time, and toc() returns
// the elapsed time since the last tic()
void tic() { gettimeofday(&g_t1, NULL); }

// Return time since last invocation of tic() in milliseconds
double toc()
{
  double ret;

  gettimeofday(&g_t2, NULL);

  ret = (g_t2.tv_sec - g_t1.tv_sec) * 1000.0;
  ret += (g_t2.tv_usec - g_t1.tv_usec) / 1000.0;
  return ret;
}

// Generate random double in range [min, max]
double rand_double(double min, double max)
{
  return min + (max - min) * ((double)rand() / (double)RAND_MAX);
}

// Simple struct to hold a square matrix
struct matrix {
  double *data;
  size_t N;
};

void matrix_init(struct matrix *m, size_t N)
{
  m->N = N;
  if ((m->data = malloc(N * N * sizeof(*m->data))) == NULL) {
    handle_error("malloc");
  }
}

// Set every element of m to a random value in the range [-1,1]
void matrix_randomize(struct matrix *m)
{
  size_t N = m->N;
  for (size_t i = 0; i < N * N; ++i) {
    m->data[i] = rand_double(-1.0, 1.0);
  }
}

// Zero-out matrix m
void matrix_zero(struct matrix *m)
{
  size_t N = m->N;
#pragma omp parallel for
  for (size_t i = 0; i < N * N; ++i) {
    m->data[i] = 0.0;
  }
}

// Naive implementation of matrix multiplication
// Computes a*b and stores result in res
// a, b, and res must all have the same dimension
void matrix_mult_naive(struct matrix *a, struct matrix *b, struct matrix *res)
{
  size_t N = a->N;
  matrix_zero(res);
#pragma omp parallel for
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      for (size_t k = 0; k < N; ++k)
        res->data[IDX(N, i, j)] +=
            a->data[IDX(N, i, k)] * b->data[IDX(N, k, j)];
}

// Cache-friendly implementation of matrix multiplication
// Computes a*b and stores result in res
// a, b, and res must all have the same dimension, and the dimension must
// be a power of 2
void matrix_mult_fast(struct matrix *a, struct matrix *b, struct matrix *res)
{
  size_t N = a->N;
  matrix_zero(res);
  size_t SM = 4 * g_linesize / sizeof(double);

  size_t i, j, k, i2, j2, k2;
  double *ra, *rb, *rres;

#pragma omp parallel for
  for (i = 0; i < N; i += SM) {
    for (j = 0; j < N; j += SM) {
      for (k = 0; k < N; k += SM) {
        for (i2 = 0, rres = &res->data[IDX(N, i, j)],
            ra = &a->data[IDX(N, i, k)];
             i2 < SM; ++i2, rres += N, ra += N) {
          for (k2 = 0, rb = &b->data[IDX(N, k, j)]; k2 < SM; ++k2, rb += N) {
#pragma GCC ivdep
            for (j2 = 0; j2 < SM; ++j2) {
              rres[j2] += ra[k2] * rb[j2];
            }
          }
        }
      }
    }
  }
}

// Return 1 if a == b, 0 otherwise
int matrix_is_equal(struct matrix *a, struct matrix *b)
{
  size_t N = a->N;
  if (b->N != N) { return 0; }

  for (size_t i = 0; i < N * N; ++i) {
    if (a->data[i] != b->data[i]) return 0;
  }

  return 1;
}

int main(int argc, char *argv[])
{

  if (argc != 2) {
    printf("Usage: %s dim\n", argv[0]);
    printf("DIM must be a power of 2\n");
    exit(EXIT_FAILURE);
  }

  errno = 0;
  size_t N = strtoul(argv[1], NULL, 0);
  if (errno) { handle_error("atoul"); }

  if (N < 2 || (N & (N - 1))) {
    printf("DIM must be >= 2 and be a power of 2\n");
    exit(EXIT_FAILURE);
  }

  printf("Matrix dimension N=%lu\n", N);

  //if ((g_linesize = sysconf(_SC_LEVEL1_DCACHE_LINESIZE)) == -1) {
  //  handle_error("sysconf");
  //}
  g_linesize = 64;
  printf("Cache line size: %ld\n", g_linesize);

  struct matrix a, b, res_naive, res_fast;

  printf("Preparing matrices ...\n");
  tic();
  matrix_init(&a, N);
  matrix_randomize(&a);
  matrix_init(&b, N);
  matrix_randomize(&b);
  matrix_init(&res_naive, N);
  matrix_init(&res_fast, N);
  printf("Done. %.3lf ms\n", toc());

  printf("Performing naive multiplication ...\n");
  tic();
  matrix_mult_naive(&a, &b, &res_naive);
  printf("Done. %.3lf ms\n", toc());

  printf("Performing fast multiplication ...\n");
  tic();
  matrix_mult_fast(&a, &b, &res_fast);
  printf("Done. %.3lf ms\n", toc());

  printf("Verifying ...\n");
  tic();
  printf("%s", matrix_is_equal(&res_naive, &res_fast) ? "PASSED" : "FAILED");
  printf(" %lf ms\n", toc());

  return 0;
}
