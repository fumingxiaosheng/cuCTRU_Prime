#include "kernel.h"
#include "params.h"
#include "reduce.h"

__device__ int16_t montgomery_reduce_int16_t(int32_t a)
{
  int32_t t;
  int16_t u;

  u = a * QINV;
  t = (int32_t)u * FPTRU_Q;
  t = a - t;
  t >>= 16;
  return t;
}

__device__ int16_t barrett_reduce_int16_t(int16_t a)
{
  int32_t t;

#if (FPTRU_LOGQ == 12)
  t = V_BARRETT * a;
  t >>= 26;
  t *= FPTRU_Q;
#elif (FPTRU_LOGQ == 13)
  t = V_BARRETT * a;
  t >>= 27;
  t *= FPTRU_Q;
#endif

  return a - t;
}

/*2024-3-30
调整a的值，在范围[0,FPTRU_Q-1)上*/
__device__ int16_t fqcsubq(int16_t a)
{
  a += (a >> 15) & FPTRU_Q;//a>>15取出a的符号位，若a为整数，则a>>15=0,否则a>>15=-1(对应的二进制表示为全1)。总的来说，若a为整数,则加上0，否则加上FPTRU_Q
  a -= FPTRU_Q;//a的值在一个以FPTRU_Q为模的周期内
  a += (a >> 15) & FPTRU_Q;//根据a的符号，调整a的值
  return a;
}

__device__ int16_t fqmul_int16_t(int16_t a, int16_t b)
{
  return montgomery_reduce_int16_t((int32_t)a * b);
}