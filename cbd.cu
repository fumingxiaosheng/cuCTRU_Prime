#include "kernel.h"

#include <iostream> //用于调试使用

//注释：is_double决定了是否需要加上1
#if (FPTRU_BOUND == 5)

/*功能:以小端方式将四个字节转化为一个32bit的数*/
__device__ uint32_t load32_littleendian(const uint8_t x[4])
{
    uint32_t r;
    r = (uint32_t)x[0];
    r |= (uint32_t)x[1] << 8;
    r |= (uint32_t)x[2] << 16;
    r |= (uint32_t)x[3] << 24;
    return r;
}

__device__ void cbd2(poly *r,const uint8_t buf[FPTRU_COIN_BYTES / 2],uint16_t is_double,uint16_t is_add1){
    unsigned int i, j;
    uint32_t t, d;
    int16_t a, b;

    i = threadIdx.x;

#if (FPTRU_N == 761)

    if(i == FPTRU_N / 8){
        t = buf[4 * i];
        a = 0;
        b = 0;
        //直接基于t生成b
        for (j = 0; j < 2; j++){
            a += (t >> j) & 0x1;
        }
        for (j = 2; j < 4; j++){
            b += (t >> j) & 0x1;
        }
        r->coeffs[8 * i + 0] = (int16_t)(a - b) + is_double * (int16_t)(a - b) + is_add1 * (!(8 * i + 0));
    }
    else{
        t = load32_littleendian(buf + 4 * i);
        d = t & 0x55555555;
        d += (t >> 1) & 0x55555555;

        for (j = 0; j < 8; j++){
            a = (d >> (4 * j + 0)) & 0x3;//
            b = (d >> (4 * j + 2)) & 0x3;
            r->coeffs[8 * i + j] = a - b +  is_double * (a - b) + is_add1 * (!(8 * i + j)); //基于d生成a和b，计算对应的值
        }
    }

#elif (FPTRU_N == 1277)
    if(i == FPTRU_N / 8){
        for (int k = 0; k < 5; k++){
            t = (buf[4 * i + (k >> 1)] >> ((k & 1) << 2)) & 0xf;
            a = 0;
            b = 0;
            for (j = 0; j < 2; j++){
                a += (t >> j) & 0x1;
            }
            for (j = 2; j < 4; j++){
                b += (t >> j) & 0x1;
            }
            r->coeffs[8 * i + k] = (int16_t)(a - b) + is_double * (int16_t)(a - b) + is_add1 * (!(8 * i + k));//最后一个生成五个系数
        }
    }
    else{
        t = load32_littleendian(buf + 4 * i);
        d = t & 0x55555555;
        d += (t >> 1) & 0x55555555;

        for (j = 0; j < 8; j++){
            a = (d >> (4 * j + 0)) & 0x3;
            b = (d >> (4 * j + 2)) & 0x3;
            r->coeffs[8 * i + j] = a - b + is_double * (a - b) + is_add1 * (!(8 * i + j));
        }
    }

#endif // end of #if (FPTRU_N == 761)
}


__device__ void cbd2_761(poly *r,const uint8_t buf[FPTRU_COIN_BYTES / 2],uint16_t is_double,uint16_t is_add1){
    unsigned int i, j;
    uint32_t t, d;
    int16_t a, b;

    j = threadIdx.x % 8;

#pragma unroll
    for(i = threadIdx.x / 8; i < 72; i+=24){
        t = load32_littleendian(buf + 4 * i);
        d = t & 0x55555555;
        d += (t >> 1) & 0x55555555;

        a = (d >> (4 * j + 0)) & 0x3;
        b = (d >> (4 * j + 2)) & 0x3;
        r->coeffs[8 * i + j] = a - b +  is_double * (a - b) + is_add1 * (!(8 * i + j));
    }

    if(threadIdx.x < 185){//761 = 192 * 3 + 185
        if(threadIdx.x == 184){
            i = 95;
            t = buf[4 * i];
            a = 0;
            b = 0;
            //直接基于t生成b
            for (j = 0; j < 2; j++){
                a += (t >> j) & 0x1;
            }
            for (j = 2; j < 4; j++){
                b += (t >> j) & 0x1;
            }
            r->coeffs[8 * i + 0] = (int16_t)(a - b) + is_double * (int16_t)(a - b) + is_add1 * (!(8 * i + 0));
        }
        else{
            t = load32_littleendian(buf + 4 * i);
            d = t & 0x55555555;
            d += (t >> 1) & 0x55555555;

            a = (d >> (4 * j + 0)) & 0x3;
            b = (d >> (4 * j + 2)) & 0x3;
            r->coeffs[8 * i + j] = a - b +  is_double * (a - b) + is_add1 * (!(8 * i + j));//int16->int32
        }
    }
}
#endif //end of #if (FPTRU_BOUND == 5)




/**********************************分割符*****************************/


#if (FPTRU_BOUND == 7)
__device__ uint64_t load_littleendian(const uint8_t *x, int bytes)
{
    int i;
    uint64_t r = x[0];
    for (i = 1; i < bytes; i++){
        r |= (uint64_t)x[i] << (8 * i);
    }
    return r;
}

__device__ void cbd3(poly *r, const uint8_t buf[FPTRU_COIN_BYTES / 2],uint16_t is_double,uint16_t is_add1){
    uint32_t t, d, a[4], b[4];
    int i, j;

    i = threadIdx.x;
#if (FPTRU_N == 653)

    if(i == FPTRU_N/4){
        t = buf[3 * i];
        a[0] = 0;
        b[0] = 0;
        for (j = 0; j < 3; j++){
            a[0] += (t >> j) & 0x1;
        }
        for (j = 3; j < 6; j++){
            b[0] += (t >> j) & 0x1;
        }
        r->coeffs[4 * i + 0] = (int16_t)(a[0] - b[0]) + is_double * (int16_t)(a[0] - b[0]) + is_add1 * (!(4 * i + 0));
    }

    else{
        t = load_littleendian(buf + 3 * i, 3);
        d = 0;
        for (j = 0; j < 3; j++){
            d += (t >> j) & 0x249249;
        }
        a[0] = d & 0x7;
        b[0] = (d >> 3) & 0x7;
        a[1] = (d >> 6) & 0x7;
        b[1] = (d >> 9) & 0x7;
        a[2] = (d >> 12) & 0x7;
        b[2] = (d >> 15) & 0x7;
        a[3] = (d >> 18) & 0x7;
        b[3] = (d >> 21);

        r->coeffs[4 * i + 0] = (int16_t)(a[0] - b[0]) + is_double * (int16_t)(a[0] - b[0]) + is_add1 * (!(4 * i + 0));
        r->coeffs[4 * i + 1] = (int16_t)(a[1] - b[1]) + is_double * (int16_t)(a[1] - b[1]) + is_add1 * (!(4 * i + 1));
        r->coeffs[4 * i + 2] = (int16_t)(a[2] - b[2]) + is_double * (int16_t)(a[2] - b[2]) + is_add1 * (!(4 * i + 2));
        r->coeffs[4 * i + 3] = (int16_t)(a[3] - b[3]) + is_double * (int16_t)(a[3] - b[3]) + is_add1 * (!(4 * i + 3));
    }

#elif (FPTRU_N == 761)
    if(i == FPTRU_N/4){
        t = buf[3 * i];
        a[0] = 0;
        b[0] = 0;
        for (j = 0; j < 3; j++){
            a[0] += (t >> j) & 0x1;
        }
        for (j = 3; j < 6; j++){
            b[0] += (t >> j) & 0x1;
        }
        r->coeffs[4 * i + 0] = (int16_t)(a[0] - b[0]) + is_double * (int16_t)(a[0] - b[0]) + is_add1 * (!(4 * i + 0));
    }
    else{
        t = load_littleendian(buf + 3 * i, 3);
        d = 0;
        for (j = 0; j < 3; j++){
            d += (t >> j) & 0x249249;
        }

        a[0] = d & 0x7;
        b[0] = (d >> 3) & 0x7;
        a[1] = (d >> 6) & 0x7;
        b[1] = (d >> 9) & 0x7;
        a[2] = (d >> 12) & 0x7;
        b[2] = (d >> 15) & 0x7;
        a[3] = (d >> 18) & 0x7;
        b[3] = (d >> 21);

        r->coeffs[4 * i + 0] = (int16_t)(a[0] - b[0]) + is_double * (int16_t)(a[0] - b[0]) + is_add1 * (!(4 * i + 0)));
        r->coeffs[4 * i + 1] = (int16_t)(a[1] - b[1]) + is_double * (int16_t)(a[1] - b[1]) + is_add1 * (!(4 * i + 1));
        r->coeffs[4 * i + 2] = (int16_t)(a[2] - b[2]) + is_double * (int16_t)(a[2] - b[2]) + is_add1 * (!(4 * i + 2));
        r->coeffs[4 * i + 3] = (int16_t)(a[3] - b[3]) + is_double * (int16_t)(a[3] - b[3]) + is_add1 * (!(4 * i + 3));   
    }
#endif // end of #if (FPTRU_N == 653)
}
#endif //end of #if (FPTRU_BOUND == 7)
