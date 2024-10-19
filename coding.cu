#include "params.h"
#include "kernel.h"
__device__ uint8_t encode_e8(uint8_t m)
{
    uint8_t a0 = m & 1;
    uint8_t a1 = (m >> 1) & 1;
    uint8_t a2 = (m >> 2) & 1;
    uint8_t a3 = (m >> 3) & 1;
    return ((-a1) & 0x0F) ^ ((-a2) & 0x3c) ^ ((-a3) & 0xF0) ^ ((-a0) & 0x55);
}