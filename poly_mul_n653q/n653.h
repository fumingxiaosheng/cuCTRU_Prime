#ifndef N653_H
#define N653_H

#include "../kernel.h"
#include "../params.h"
#include "../poly.h"


#define N_N653 1344
#define Q_N653 16777153//伪梅森数环的Q
#define NUMBER_MZXJDUDN 16128
#define NUMBER_ACONFI 3221491777 //Q_N653在2^32上的逆
#define NUMBER_AKNCF 16761025 //TODO:这个宏的含义是什么->应该代表的是旋转因子为1
#define FACTOR_CNSHHUI 8197629
#define FACTOR_NAKYFHAIRNF 1354752


typedef struct
{
    int32_t coeffs[N_N653];
} nttpoly_n653;

void poly_mul_653_batch_q1(poly *array_c,poly * array_a,poly *array_b,cudaStream_t stream,int batch_size);

__global__ void poly_basemul_n653_batch(nttpoly_n653 *array_c, nttpoly_n653 *array_a, nttpoly_n653 *array_b);

__global__ void poly_extend_n653_batch(nttpoly_n653 * array_nttb,poly * array_a);

__global__ void poly_ntt_big_n653_batch(nttpoly_n653 * array_a);

__global__ void poly_ntt_small_n653_batch(nttpoly_n653 * array_a);

__global__ void poly_invntt_n653_batch(nttpoly_n653 * array_a);

__device__ void poly_extend_n653(nttpoly_n653 * b,poly * a);

__device__ void ntt_big_n653(int32_t a[N_N653]);

__device__ void ntt_small_n653(int32_t a[N_N653]);

__device__ void poly_basemul_n653(nttpoly_n653 *c, nttpoly_n653 *a, nttpoly_n653 *b);

__device__ void basemul_n653(int32_t c[7], const int32_t a[7], const int32_t b[7], const int32_t zeta);

__device__ void invntt_tomont_n653(int32_t a[N_N653]);

__device__ void poly_extract_n653_q1(poly *b,nttpoly_n653 *a);

__global__ void poly_extract_n653_q1_batch(poly * array_a ,nttpoly_n653 * array_ntta);

__global__ void poly_mul_653_batch_q1_v2(poly *array_c,poly * array_a,poly *array_b);

__device__ void poly_mul_653_batch_q1_v2_device(poly src_c, poly src_a,poly *array_b);

__device__ void poly_mul_653_batch_q1_v2_device_v2(poly*array_c, poly* array_a,poly *array_b);

__global__ void poly_mul_653_batch_q1_v2_test(poly *array_c,poly * array_a,poly *array_b,nttpoly_n653 * array_ntta,nttpoly_n653 * array_nttb);

__device__ poly poly_mul_653_batch_q1_v2_device_test(poly * array_a,poly *array_b);

__global__ void poly_mul_653_batch_q1_zhc(poly *array_c,poly * array_a,poly *array_b);

__device__ void ntt_big_n653_v2(int32_t a[N_N653]);

__global__ void poly_mul_653_batch_q1_v3(poly *array_c,poly * array_a,poly *array_b);

__device__ void poly_mul_653_batch_q1_v3_device(poly *array_c,poly * array_a,poly *array_b);

__device__ int32_t montgomery_reduce_n653_ptx(int64_t a);

__device__ int32_t montgomery_reduce_n653_cuda(int64_t a);

__global__ void poly_mul_653_batch_q2(poly *array_c,poly * array_a,poly *array_b);
#endif