#ifndef PACK_H
#define PACK_H


#include <stdint.h>
#include "poly.h"
#include "kernel.h"
#include "params.h"

__global__ void pack_pk_batch(unsigned char *array_r,poly * array_a);

__global__ void pack_sk_batch(unsigned char *array_r,poly *array_a);

__device__ void pack_sk_v2(unsigned char * r,const poly *a);

__global__ void pack_sk_batch_v2(unsigned char *array_r, poly *array_a,int interval);

void pack_pk(unsigned char *r, poly *a);
void pack_sk(unsigned char *r, poly *a);

void unpack_pk(poly *r, const unsigned char *a);

__global__ void pack_ct_batch(unsigned char *r, const poly *a);

void unpack_ct(poly *r, const unsigned char *a);

void unpack_sk(poly *r, const unsigned char *a);

__global__ void unpack_ct_batch(poly * r,const unsigned char *a);

__global__ void poly_fqcsubq_encode_compress_batch_pack_ct_653(unsigned char *r, poly *array_sigma, unsigned char *arrar_msg, int interval);

__global__ void poly_fqcsubq_encode_compress_batch_pack_ct_761(unsigned char *r, poly *array_sigma, unsigned char *arrar_msg, int interval);

__global__ void poly_fqcsubq_encode_compress_batch_pack_ct_1277(unsigned char *r, poly *array_sigma, unsigned char *arrar_msg, int interval);

__global__ void unpack_pk_batch(poly *r, unsigned char *a);

__global__ void pack_pk_simple_batch(unsigned char *r_pk,unsigned char *r_sk, const poly *a);

__global__ void unpack_pk_simple_batch(poly *r, const unsigned char *a, int interval);

__global__ void unpack_sk_batch(poly *r, const unsigned char *a);
#endif