#ifndef TENSOR_H
#define TENSOR_H

#include"kernel.h"
void tensor_poly_mul(poly *c, poly *a,poly *b,cudaStream_t stream);
void tensor_poly_mul_v2(poly *c, poly *a,poly *b,cudaStream_t stream);
void tensor_poly_mul_v3(poly *c, poly *a,poly *b,cudaStream_t stream);
#define TC_X 16
#define TC_Y 16
#define TC_Z 16


#define A_x 656
#define A_y 656

#define B_x 656
#define B_y 16

#define N 653
#define Q 4621

#endif