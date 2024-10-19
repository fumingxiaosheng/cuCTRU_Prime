#ifndef CBD_H
#define CBD_H

#include "params.h"
#include "kernel.h"
__device__ void cbd2(poly *r,const uint8_t buf[FPTRU_COIN_BYTES / 2],uint16_t is_double,uint16_t is_add1);
__device__ void cbd3(poly *r, const uint8_t buf[FPTRU_COIN_BYTES / 2],uint16_t is_double,uint16_t is_add1);
__device__ void cbd2_761(poly *r,const uint8_t buf[FPTRU_COIN_BYTES / 2],uint16_t is_double,uint16_t is_add1);
#if (FPTRU_BOUND == 5)
__device__ uint32_t load32_littleendian(const uint8_t x[4]);//remove static
#endif
#if (FPTRU_BOUND == 7)
__device__ uint64_t load_littleendian(const uint8_t *x, int bytes);
#endif
#endif
