#ifndef POLY_H
#define POLY_H
#include "kernel.h"


__global__ void poly_sample(poly *array_a,unsigned char * array_coins);

__global__ void poly_double(poly *b, poly *a);

void poly_inverse_batch_0(poly * array_finv,poly * array_f,cudaStream_t stream);

__device__ int16_t fq_freeze(int32_t x);

__global__ void poly_fqcsubq_batch(poly *array_a);

__device__ uint16_t uint32_mod_uint14(uint32_t x, uint16_t m);

__device__ void uint32_divmod_uint14(uint32_t *y, uint16_t *r, uint32_t x, uint16_t m);


__global__ void poly_inverse_once(poly * finv, poly * f);
__device__ int rq_inverse(int16_t *finv, const int16_t *f);
__device__ void poly_inverse(poly *b, const poly *a);

__global__ void poly_sample_and_double_v2(poly *array_a,unsigned char * array_coins,uint16_t is_double,uint16_t is_add1,int interval);

__device__ void poly_double_v2(poly *array_a,poly a);

void poly_inverse_batch_0_v2(poly * array_finv,poly * array_f,cudaStream_t stream);

//__global__ void N_poly_mul(poly * res_fmul, poly * src_array_f);

__global__ void N_poly_mul(poly * res_fmul, poly * array_f);

__device__ void rq_inverse_v2(int16_t *finv,const int16_t *f);

__global__ void N_inverse_recover(poly * src_fmul_inv,poly * array_f ,poly * array_finv);

__global__ void poly_fqcsubq_batch_v2(poly *array_a);

uint16_t uint32_mod_uint14_cpu(uint32_t x, uint16_t m);

void uint32_divmod_uint14_cpu(uint32_t *y, uint16_t *r, uint32_t x, uint16_t m);

__device__ void poly_mul_653_batch_q1_v2_device(poly *array_c,poly * array_a,poly *array_b);

void poly_inverse_batch_0_v3(poly * array_finv,poly * array_f,cudaStream_t stream);

__global__ void N_poly_mul_v3(poly * res_fmul, poly * array_f);

__global__ void N_inverse_recover_v3(poly * src_fmul_inv,poly * array_f ,poly * array_finv);

__global__ void poly_inv(poly * finv, poly * f);

__device__ int16_t fq_freeze(int32_t x);

__global__ void poly_fqcsubq_encode_compress_batch(poly * c, poly * sigma, unsigned char * msg,int interval);

__device__ void rq_inverse_v4(int16_t *finv,const int16_t *f);

__device__ void rq_inverse_v3(int16_t *finv,const int16_t *f);

__device__ void rq_inverse_clean(int16_t *finv,const int16_t *f);

__global__ void poly_decode_batch(unsigned char *m, poly *mp);

__global__ void poly_inv_1277(poly * finv, poly * f);

__global__ void poly_fqcsubq_batch_1277(poly *array_a);
#endif