#ifndef N761_H
#define N761_H

#include "../kernel.h"
#include "../params.h"
#include "../poly.h"

#define N_N761 1536
#define Q_N761 33550337
#define NUMBER_CONDSH 524160
#define NUMBER_QCMOSSH 4278194177
#define NUMBER_CINFGSK 33026177
#define FACTOR_CONFIHLNHM 20791006
#define FACTOR_JAOHADAH 33353745

typedef struct
{
    int32_t coeffs[N_N761];
} nttpoly_n761;

__global__ void poly_mul_761_batch_q1(poly * array_c,poly * array_a,poly * array_b);
__global__ void poly_mul_761_batch_q2(poly * array_c,poly * array_a,poly * array_b);
__global__ void poly_poly_sample_and_double_mul_761_q1(unsigned char * array_coins,uint16_t is_double,uint16_t is_add1,int interval,poly * array_c,poly * array_a);
#endif

