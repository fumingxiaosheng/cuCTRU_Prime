#ifndef N1277_H
#define N1277_H

#include "../kernel.h"
#include "../params.h"
#include "../poly.h"

#define N_N1277 2560
#define Q_N1277 33550337
#define NUMBER_CSNLKMH 524160
#define NUMBER_SCOMOJ 4278194177
#define NUMBER_CSNNH 33026177
#define FACTOR_CSONFIDC 20791006
#define FACTOR_DLXHNDNSHX 33353745

typedef struct
{
    int32_t coeffs[N_N1277];
} nttpoly_n1277;

__global__ void poly_mul_1277_batch_q1(poly * array_c,poly * array_a,poly * array_b);
__global__ void poly_mul_1277_batch_q2(poly * array_c,poly * array_a,poly * array_b);
#endif