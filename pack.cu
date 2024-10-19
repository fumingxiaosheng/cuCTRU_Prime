#include "kernel.h"
#include "params.h"
#include "poly.h"
#include "pack.h"

#include <iostream>

#define RM_LEN 1277
#if (USING_PK_ENCODE == 1)
/* Based on the reference implementation of NTRU Prime (NIST 3rd round submission)
 * by Daniel J. Bernstein, Chitchanok Chuengsatiansup, Tanja Lange, Christine van Vredendaal.
 * It can be used for q \in {4091, 4621, 4591, 7879}.
 * */

void Encode(unsigned char *out, const uint16_t *R, const uint16_t *M, const long long len)
{
    if (len == 1)
    {
        uint16_t r = R[0];
        uint16_t m = M[0];
        while (m > 1)
        {
            *out++ = r;
            r >>= 8;
            m = (m + 255) >> 8;
        }
    }

    if (len > 1)
    {
        uint16_t R2[(len + 1) / 2];
        uint16_t M2[(len + 1) / 2];
        long long i;

        for (i = 0; i < len - 1; i += 2)
        {
            uint32_t m0 = M[i];
            uint32_t r = R[i] + R[i + 1] * m0;
            uint32_t m = M[i + 1] * m0;

            while (m >= 16384)
            {
                *out++ = r; //拼接操作限制了多个线程的并发
                r >>= 8;
                m = (m + 255) >> 8;
            }
            R2[i / 2] = r;
            M2[i / 2] = m;
        }

        if (i < len)
        {
            R2[i / 2] = R[i];
            M2[i / 2] = M[i];
        }

        Encode(out, R2, M2, (len + 1) / 2);
    }
}

void pack_pk(unsigned char *r, poly *a)
{
    uint16_t R[FPTRU_N], M[FPTRU_N];

    for (int i = 0; i < FPTRU_N; ++i)
        R[i] = (uint16_t)a->coeffs[i];

    for (int i = 0; i < FPTRU_N; ++i)
        M[i] = FPTRU_Q;

    Encode(r, R, M, FPTRU_N);
}

__device__ void Encode_2_gpu(unsigned char *out,uint16_t *R, uint16_t *M, long long len){
    uint16_t R2[RM_LEN];
    uint16_t M2[RM_LEN];
    long long i;//注意是否需要放到循环内部
    for(;len!=1;len=(len + 1) / 2){
        for(i=0;i<len-1;i+=2){
            uint32_t m0=M[i];
            uint32_t r=R[i]+R[i+1]*m0;
            uint32_t m=M[i+1]*m0;
            while (m >= 16384)
            {
                *out++ = r;//TODO:这里应该如何实现为线程的同步:1.最外层接->不行
                r >>= 8;
                m = (m + 255) >> 8;
            }
            R2[i / 2] = r;
            M2[i / 2] = m;
        }

        if (i < len){
            R2[i / 2] = R[i];//直接把奇数的值赋值过去
            M2[i / 2] = M[i];
        }

        //递归调用的过程，本质上是转化了R和R2的位置
        for(int j=0;j<RM_LEN;j++){
            R[j]=R2[j];
            M[j]=M2[j];
        }
    }

    if (len == 1)
    {
        uint16_t r = R[0];
        uint16_t m = M[0];
        while (m > 1)
        {
            *out++ = r;
            r >>= 8;
            m = (m + 255) >> 8;
        }
    }

}
__device__ void pack_pk_gpu(unsigned char *r, const poly *a)
{   
    //printf("[in pack_pk] start\n");
    //uint16_t R[FPTRU_N], M[FPTRU_N];
    uint16_t R[RM_LEN], M[RM_LEN];

    for (int i = 0; i < FPTRU_N; ++i)
        R[i] = (uint16_t)a->coeffs[i];

    for (int i = 0; i < FPTRU_N; ++i)
        M[i] = FPTRU_Q;
    
    
    Encode_2_gpu(r, R, M, FPTRU_N);
}

/*2024-4-24:
输入:array_r:待处理的公钥数组
    array_a:待转化的多项式数组

处理流程:数据分组，分给每一个线程进行处理

线程组织:<<<BATCH_SIZE,1>>>
*/
__global__ void pack_pk_batch(unsigned char *array_r,poly * array_a){
    //printf("[%d %d] start\n",blockIdx.x,threadIdx.x);
    pack_pk_gpu(&array_r[FPTRU_KEM_PUBLICKEYBYTES * (blockIdx.x)], &(array_a[blockIdx.x]));
    //printf("[%d %d] end\n",blockIdx.x,threadIdx.x);
}
#endif



#if (USING_PK_ENCODE == 1)
void Decode(uint16_t *out, const unsigned char *S, const uint16_t *M, const long long len)
{
    if (len == 1)
    {
        if (M[0] == 1)
            *out = 0;
        else if (M[0] <= 256)
            *out = uint32_mod_uint14_cpu(S[0], M[0]);
        else
            *out = uint32_mod_uint14_cpu(S[0] + (((uint16_t)S[1]) << 8), M[0]);
    }

    if (len > 1)
    {
        uint16_t R2[(len + 1) / 2];
        uint16_t M2[(len + 1) / 2];
        uint16_t bottomr[len / 2];
        uint32_t bottomt[len / 2];


        // uint16_t R2[1277];//uint16_t R2[(len + 1) / 2];
        // uint16_t M2[1277];//uint16_t M2[(len + 1) / 2];
        // uint16_t bottomr[1277];//uint16_t bottomr[len / 2];
        // uint32_t bottomt[1277];//uint32_t bottomt[len / 2];
        long long i;
        for (i = 0; i < len - 1; i += 2)
        {
            uint32_t m = M[i] * (uint32_t)M[i + 1];
            if (m > 256 * 16383)
            {
                bottomt[i / 2] = 256 * 256;
                bottomr[i / 2] = S[0] + 256 * S[1];
                S += 2;
                M2[i / 2] = (((m + 255) >> 8) + 255) >> 8;
            }
            else if (m >= 16384)
            {
                bottomt[i / 2] = 256;
                bottomr[i / 2] = S[0];
                S += 1;
                M2[i / 2] = (m + 255) >> 8;
            }
            else
            {
                bottomt[i / 2] = 1;
                bottomr[i / 2] = 0;
                M2[i / 2] = m;
            }
        }
        if (i < len)
            M2[i / 2] = M[i];

        Decode(R2, S, M2, (len + 1) / 2);

        for (i = 0; i < len - 1; i += 2)
        {
            uint32_t r = bottomr[i / 2];//和当时自己的临时状态有关系
            uint32_t r1;
            uint16_t r0;
            r += bottomt[i / 2] * R2[i / 2]; //当前的输入和上一级的输出有关系
            uint32_divmod_uint14_cpu(&r1, &r0, r, M[i]);
            r1 = uint32_mod_uint14_cpu(r1, M[i + 1]);
            *out++ = r0;
            *out++ = r1;
        }
        if (i < len)
            *out++ = R2[i / 2];
    }
}

void unpack_pk(poly *r, const unsigned char *a)
{
    uint16_t R[FPTRU_N], M[FPTRU_N];

    for (int i = 0; i < FPTRU_N; ++i)
        M[i] = FPTRU_Q;

    Decode(R, a, M, FPTRU_N);

    for (int i = 0; i < FPTRU_N; ++i)
        r->coeffs[i] = ((int16_t)R[i]);
}

/*__global__ void unpack_pk_batch(){

}*/
#endif



void pack_pk_simple(unsigned char *r, const poly *a)
{
    unsigned int i = 0 ;
    for (i = 0; i < FPTRU_N/8; i ++) //95
    {
        r[i *13] = a->coeffs[i*8];
        r[i *13+ 1] = (a->coeffs[i*8] >> 8)| ((int16_t)a->coeffs[i*8+1] << 5);
        r[i *13+ 2] = a->coeffs[i*8+1] >> 3;
        r[i *13+ 3] = (a->coeffs[i*8+1] >> 11 )| ((int16_t)a->coeffs[i*8+2] << 2);
        r[i *13+ 4] = (a->coeffs[i*8+2] >> 6  )| ((int16_t)a->coeffs[i*8+3] << 7);
        r[i *13+ 5] = a->coeffs[i*8+3] >> 1;
        r[i *13+ 6] = (a->coeffs[i*8+3] >> 9  )| ((int16_t)a->coeffs[i*8+4] << 4);
        r[i *13+ 7] = a->coeffs[i*8+4] >> 4;
        r[i *13+ 8] = (a->coeffs[i*8+4] >> 12 )| ((int16_t)a->coeffs[i*8+5] << 1);
        r[i *13+ 9] = (a->coeffs[i*8+5] >> 7  )| ((int16_t)a->coeffs[i*8+6] << 6);
        r[i *13+ 10] = a->coeffs[i*8+6] >> 2;
        r[i *13+ 11] = (a->coeffs[i*8+6] >> 10) | ((int16_t)a->coeffs[i*8+7] << 3);
        r[i *13+ 12] = a->coeffs[i*8+7] >> 5;
    }
    r[i*13] = a->coeffs[i*8];
    r[i*13+ 1] = (a->coeffs[i*8] >> 8) & 0x1F;
}

__device__ void pack_pk_simple_gpu(unsigned char *r, const poly *a){
    int i = threadIdx.x;
    if(i == FPTRU_N/8){
        r[i*13] = a->coeffs[i*8];
        r[i*13+ 1] = (a->coeffs[i*8] >> 8) & 0x1F;
    }
    else{
        r[i *13] = a->coeffs[i*8];
        r[i *13+ 1] = (a->coeffs[i*8] >> 8)| ((int16_t)a->coeffs[i*8+1] << 5);
        r[i *13+ 2] = a->coeffs[i*8+1] >> 3;
        r[i *13+ 3] = (a->coeffs[i*8+1] >> 11 )| ((int16_t)a->coeffs[i*8+2] << 2);
        r[i *13+ 4] = (a->coeffs[i*8+2] >> 6  )| ((int16_t)a->coeffs[i*8+3] << 7);
        r[i *13+ 5] = a->coeffs[i*8+3] >> 1;
        r[i *13+ 6] = (a->coeffs[i*8+3] >> 9  )| ((int16_t)a->coeffs[i*8+4] << 4);
        r[i *13+ 7] = a->coeffs[i*8+4] >> 4;
        r[i *13+ 8] = (a->coeffs[i*8+4] >> 12 )| ((int16_t)a->coeffs[i*8+5] << 1);
        r[i *13+ 9] = (a->coeffs[i*8+5] >> 7  )| ((int16_t)a->coeffs[i*8+6] << 6);
        r[i *13+ 10] = a->coeffs[i*8+6] >> 2;
        r[i *13+ 11] = (a->coeffs[i*8+6] >> 10) | ((int16_t)a->coeffs[i*8+7] << 3);
        r[i *13+ 12] = a->coeffs[i*8+7] >> 5;
    }

}
__global__ void pack_pk_simple_batch(unsigned char *r, const poly *a){
    pack_pk_simple_gpu(&r[FPTRU_KEM_PUBLICKEYBYTES * blockIdx.x], &a[blockIdx.x]);
}



void unpack_pk_simple(poly *r, const unsigned char *a) {

    unsigned int i = 0;
    for (i = 0; i < FPTRU_N/8; i++) {
        r->coeffs[i*8] = a[i*13] | (uint16_t)a[i*13 + 1] << 8;
        r->coeffs[i*8] &= 0x1FFF;
        r->coeffs[i*8+1] = (a[i*13 + 1] >> 5) | (uint16_t)(a[i*13 + 2]  << 3) | (uint16_t)(a[i*13 + 3]  << 11) ;
        r->coeffs[i*8+1] &= 0x1FFF;
        r->coeffs[i*8+2] = (a[i*13 + 3] >> 2) | (uint16_t)(a[i*13 + 4]  << 6);
        r->coeffs[i*8+2] &= 0x1FFF;
        r->coeffs[i*8+3] = (a[i*13 + 4] >> 7) | (uint16_t)(a[i*13 + 5]  << 1)| (uint16_t)(a[i*13 + 6]   << 9);
        r->coeffs[i*8+3] &= 0x1FFF;
        r->coeffs[i*8+4] = (a[i*13 + 6] >> 4) | (uint16_t)(a[i*13 + 7]  << 4) | (uint16_t)(a[i*13 + 8]  << 12);
        r->coeffs[i*8+4] &= 0x1FFF;
        r->coeffs[i*8+5] = (a[i*13 + 8] >> 1) | (uint16_t)(a[i*13 + 9]  << 7) ;
        r->coeffs[i*8+5] &= 0x1FFF;
        r->coeffs[i*8+6] = (a[i*13 + 9] >> 6) | (uint16_t)(a[i*13 + 10] << 2) | (uint16_t)(a[i*13 + 11] << 10);
        r->coeffs[i*8+6] &= 0x1FFF;
        r->coeffs[i*8+7] = (a[i*13 + 11] >> 3) | (uint16_t)(a[i*13 + 12] << 5) ;
        r->coeffs[i*8+7] &= 0x1FFF;
    }
    r->coeffs[i*8] = a[i*13] | (uint16_t)a[i*13 + 1] << 8;
    r->coeffs[i*8] &= 0x1FFF;
}

__device__ void unpack_pk_simple_gpu(poly *r, const unsigned char *a){
    int i = threadIdx.x;
    if(i == FPTRU_N/8){
        r->coeffs[i*8] = a[i*13] | (uint16_t)a[i*13 + 1] << 8;
        r->coeffs[i*8] &= 0x1FFF;
    }
    else{
        r->coeffs[i*8] = a[i*13] | (uint16_t)a[i*13 + 1] << 8;
        r->coeffs[i*8] &= 0x1FFF;
        r->coeffs[i*8+1] = (a[i*13 + 1] >> 5) | (uint16_t)(a[i*13 + 2]  << 3) | (uint16_t)(a[i*13 + 3]  << 11) ;
        r->coeffs[i*8+1] &= 0x1FFF;
        r->coeffs[i*8+2] = (a[i*13 + 3] >> 2) | (uint16_t)(a[i*13 + 4]  << 6);
        r->coeffs[i*8+2] &= 0x1FFF;
        r->coeffs[i*8+3] = (a[i*13 + 4] >> 7) | (uint16_t)(a[i*13 + 5]  << 1)| (uint16_t)(a[i*13 + 6]   << 9);
        r->coeffs[i*8+3] &= 0x1FFF;
        r->coeffs[i*8+4] = (a[i*13 + 6] >> 4) | (uint16_t)(a[i*13 + 7]  << 4) | (uint16_t)(a[i*13 + 8]  << 12);
        r->coeffs[i*8+4] &= 0x1FFF;
        r->coeffs[i*8+5] = (a[i*13 + 8] >> 1) | (uint16_t)(a[i*13 + 9]  << 7) ;
        r->coeffs[i*8+5] &= 0x1FFF;
        r->coeffs[i*8+6] = (a[i*13 + 9] >> 6) | (uint16_t)(a[i*13 + 10] << 2) | (uint16_t)(a[i*13 + 11] << 10);
        r->coeffs[i*8+6] &= 0x1FFF;
        r->coeffs[i*8+7] = (a[i*13 + 11] >> 3) | (uint16_t)(a[i*13 + 12] << 5) ;
        r->coeffs[i*8+7] &= 0x1FFF;
    }
}
__global__ void unpack_pk_simple_batch(poly *r, const unsigned char *a, int interval){
    unpack_pk_simple_gpu(&r[blockIdx.x], &a[blockIdx.x * interval]);
}


__device__ void pack_sk_gpu(unsigned char *r, const poly *a)
{
    unsigned int i;
    uint8_t t[8];
#if ((FPTRU_N == 653) || (FPTRU_N == 1277))
    for (i = 0; i < FPTRU_N / 8; i++)
    {
        t[0] = FPTRU_BOUND - a->coeffs[8 * i + 0];
        t[1] = FPTRU_BOUND - a->coeffs[8 * i + 1];
        t[2] = FPTRU_BOUND - a->coeffs[8 * i + 2];
        t[3] = FPTRU_BOUND - a->coeffs[8 * i + 3];
        t[4] = FPTRU_BOUND - a->coeffs[8 * i + 4];
        t[5] = FPTRU_BOUND - a->coeffs[8 * i + 5];
        t[6] = FPTRU_BOUND - a->coeffs[8 * i + 6];
        t[7] = FPTRU_BOUND - a->coeffs[8 * i + 7];

        r[4 * i + 0] = (t[0] >> 0) | (t[1] << 4);
        r[4 * i + 1] = (t[2] >> 0) | (t[3] << 4);
        r[4 * i + 2] = (t[4] >> 0) | (t[5] << 4);
        r[4 * i + 3] = (t[6] << 0) | (t[7] << 4);
    }
    t[0] = FPTRU_BOUND - a->coeffs[8 * i + 0];
    t[1] = FPTRU_BOUND - a->coeffs[8 * i + 1];
    t[2] = FPTRU_BOUND - a->coeffs[8 * i + 2];
    t[3] = FPTRU_BOUND - a->coeffs[8 * i + 3];
    t[4] = FPTRU_BOUND - a->coeffs[8 * i + 4];
    r[4 * i + 0] = (t[0] >> 0) | (t[1] << 4);
    r[4 * i + 1] = (t[2] >> 0) | (t[3] << 4);
    r[4 * i + 2] = (t[4] >> 0) & 0xf;
#elif (FPTRU_N == 761)
    for (i = 0; i < FPTRU_N / 8; i++)
    {
        t[0] = FPTRU_BOUND - a->coeffs[8 * i + 0];
        t[1] = FPTRU_BOUND - a->coeffs[8 * i + 1];
        t[2] = FPTRU_BOUND - a->coeffs[8 * i + 2];
        t[3] = FPTRU_BOUND - a->coeffs[8 * i + 3];
        t[4] = FPTRU_BOUND - a->coeffs[8 * i + 4];
        t[5] = FPTRU_BOUND - a->coeffs[8 * i + 5];
        t[6] = FPTRU_BOUND - a->coeffs[8 * i + 6];
        t[7] = FPTRU_BOUND - a->coeffs[8 * i + 7];

        r[4 * i + 0] = (t[0] >> 0) | (t[1] << 4);
        r[4 * i + 1] = (t[2] >> 0) | (t[3] << 4);
        r[4 * i + 2] = (t[4] >> 0) | (t[5] << 4);
        r[4 * i + 3] = (t[6] << 0) | (t[7] << 4);
    }
    t[0] = FPTRU_BOUND - a->coeffs[8 * i + 0];
    r[4 * i + 0] = (t[0] >> 0) & 0xf;
#endif
}

void pack_sk(unsigned char *r,poly *a)
{
    unsigned int i;
    uint8_t t[8];
#if ((FPTRU_N == 653) || (FPTRU_N == 1277))
    for (i = 0; i < FPTRU_N / 8; i++)
    {
        t[0] = FPTRU_BOUND - a->coeffs[8 * i + 0];
        t[1] = FPTRU_BOUND - a->coeffs[8 * i + 1];
        t[2] = FPTRU_BOUND - a->coeffs[8 * i + 2];
        t[3] = FPTRU_BOUND - a->coeffs[8 * i + 3];
        t[4] = FPTRU_BOUND - a->coeffs[8 * i + 4];
        t[5] = FPTRU_BOUND - a->coeffs[8 * i + 5];
        t[6] = FPTRU_BOUND - a->coeffs[8 * i + 6];
        t[7] = FPTRU_BOUND - a->coeffs[8 * i + 7];

        r[4 * i + 0] = (t[0] >> 0) | (t[1] << 4);
        r[4 * i + 1] = (t[2] >> 0) | (t[3] << 4);
        r[4 * i + 2] = (t[4] >> 0) | (t[5] << 4);
        r[4 * i + 3] = (t[6] << 0) | (t[7] << 4);
    }
    t[0] = FPTRU_BOUND - a->coeffs[8 * i + 0];
    t[1] = FPTRU_BOUND - a->coeffs[8 * i + 1];
    t[2] = FPTRU_BOUND - a->coeffs[8 * i + 2];
    t[3] = FPTRU_BOUND - a->coeffs[8 * i + 3];
    t[4] = FPTRU_BOUND - a->coeffs[8 * i + 4];
    r[4 * i + 0] = (t[0] >> 0) | (t[1] << 4);
    r[4 * i + 1] = (t[2] >> 0) | (t[3] << 4);
    r[4 * i + 2] = (t[4] >> 0) & 0xf;
#elif (FPTRU_N == 761)
    for (i = 0; i < FPTRU_N / 8; i++)
    {
        t[0] = FPTRU_BOUND - a->coeffs[8 * i + 0];
        t[1] = FPTRU_BOUND - a->coeffs[8 * i + 1];
        t[2] = FPTRU_BOUND - a->coeffs[8 * i + 2];
        t[3] = FPTRU_BOUND - a->coeffs[8 * i + 3];
        t[4] = FPTRU_BOUND - a->coeffs[8 * i + 4];
        t[5] = FPTRU_BOUND - a->coeffs[8 * i + 5];
        t[6] = FPTRU_BOUND - a->coeffs[8 * i + 6];
        t[7] = FPTRU_BOUND - a->coeffs[8 * i + 7];

        r[4 * i + 0] = (t[0] >> 0) | (t[1] << 4);
        r[4 * i + 1] = (t[2] >> 0) | (t[3] << 4);
        r[4 * i + 2] = (t[4] >> 0) | (t[5] << 4);
        r[4 * i + 3] = (t[6] << 0) | (t[7] << 4);
    }
    t[0] = FPTRU_BOUND - a->coeffs[8 * i + 0];
    r[4 * i + 0] = (t[0] >> 0) & 0xf;
#endif
}


/*2024-4-29:
输入:array_r:输出数组，用于存储每一个私钥多项式对应的输出
    array_a:输入的多项式数组，存储每一个私钥
    
输出:存储到array_r中

处理流程:对数据进行分块处理,然后再调用pack_sk_v2

线程组织形式:<<<BATCH_SIZE,FPTRU_N/8 + 1>>>*/
__global__ void pack_sk_batch_v2(unsigned char *array_r, poly *array_a){

    pack_sk_v2(&array_r[FPTRU_PKE_SECRETKEYBYTES * (blockIdx.x)], &(array_a[blockIdx.x]));
}

/*2024-4-29:
输入:r:全局内存，用于存储输出的结果
    a:全局内存，代表了待pack的私钥多项式
    
输出:存储到r中

线程组织形式:<<<BATCH_SIZE,FPTRU_N/8 + 1>>>*/
__device__ void pack_sk_v2(unsigned char * r,const poly *a){
    uint8_t t[8];//每一个线程的寄存器内存

#if ((FPTRU_N == 653) || (FPTRU_N == 1277))
    if(threadIdx.x < FPTRU_N / 8){
        t[0] = FPTRU_BOUND - a->coeffs[8 * threadIdx.x + 0];
        t[1] = FPTRU_BOUND - a->coeffs[8 * threadIdx.x + 1];
        t[2] = FPTRU_BOUND - a->coeffs[8 * threadIdx.x + 2];
        t[3] = FPTRU_BOUND - a->coeffs[8 * threadIdx.x + 3];
        t[4] = FPTRU_BOUND - a->coeffs[8 * threadIdx.x + 4];
        t[5] = FPTRU_BOUND - a->coeffs[8 * threadIdx.x + 5];
        t[6] = FPTRU_BOUND - a->coeffs[8 * threadIdx.x + 6];
        t[7] = FPTRU_BOUND - a->coeffs[8 * threadIdx.x + 7];

        r[4 * threadIdx.x + 0] = (t[0] >> 0) | (t[1] << 4);
        r[4 * threadIdx.x + 1] = (t[2] >> 0) | (t[3] << 4);
        r[4 * threadIdx.x + 2] = (t[4] >> 0) | (t[5] << 4);
        r[4 * threadIdx.x + 3] = (t[6] << 0) | (t[7] << 4);
    }
    else{
        t[0] = FPTRU_BOUND - a->coeffs[8 * threadIdx.x + 0];
        t[1] = FPTRU_BOUND - a->coeffs[8 * threadIdx.x + 1];
        t[2] = FPTRU_BOUND - a->coeffs[8 * threadIdx.x + 2];
        t[3] = FPTRU_BOUND - a->coeffs[8 * threadIdx.x + 3];
        t[4] = FPTRU_BOUND - a->coeffs[8 * threadIdx.x + 4];
        r[4 * threadIdx.x + 0] = (t[0] >> 0) | (t[1] << 4);
        r[4 * threadIdx.x + 1] = (t[2] >> 0) | (t[3] << 4);
        r[4 * threadIdx.x + 2] = (t[4] >> 0) & 0xf;
    }
#elif(FPTRU_N == 761)
    if(threadIdx.x < FPTRU_N / 8){
        t[0] = FPTRU_BOUND - a->coeffs[8 * threadIdx.x + 0];
        t[1] = FPTRU_BOUND - a->coeffs[8 * threadIdx.x + 1];
        t[2] = FPTRU_BOUND - a->coeffs[8 * threadIdx.x + 2];
        t[3] = FPTRU_BOUND - a->coeffs[8 * threadIdx.x + 3];
        t[4] = FPTRU_BOUND - a->coeffs[8 * threadIdx.x + 4];
        t[5] = FPTRU_BOUND - a->coeffs[8 * threadIdx.x + 5];
        t[6] = FPTRU_BOUND - a->coeffs[8 * threadIdx.x + 6];
        t[7] = FPTRU_BOUND - a->coeffs[8 * threadIdx.x + 7];

        r[4 * threadIdx.x + 0] = (t[0] >> 0) | (t[1] << 4);
        r[4 * threadIdx.x + 1] = (t[2] >> 0) | (t[3] << 4);
        r[4 * threadIdx.x + 2] = (t[4] >> 0) | (t[5] << 4);
        r[4 * threadIdx.x + 3] = (t[6] << 0) | (t[7] << 4);
    }
    else{
        t[0] = FPTRU_BOUND - a->coeffs[8 * threadIdx.x + 0];
        r[4 * threadIdx.x + 0] = (t[0] >> 0) & 0xf;
    }
#endif
}
/*2024-4-24:
输入:array_r:待压缩的公钥存储数组
    array_a:待压缩的多项式数组

处理流程:数据分组，分给每一个线程进行处理

线程组织:<<<BATCH_SIZE,1>>>
*/
__global__ void pack_sk_batch(unsigned char *array_r, poly *array_a){

    pack_sk_gpu(&array_r[FPTRU_PKE_SECRETKEYBYTES * (blockIdx.x)], &(array_a[blockIdx.x]));
}

void unpack_sk(poly *r, const unsigned char *a)
{
    unsigned int i;
#if ((FPTRU_N == 653) || (FPTRU_N == 1277))
    for (i = 0; i < FPTRU_N / 8; ++i)
    {
        r->coeffs[8 * i + 0] = (a[4 * i + 0] >> 0) & 15;
        r->coeffs[8 * i + 1] = (a[4 * i + 0] >> 4) & 15;
        r->coeffs[8 * i + 2] = (a[4 * i + 1] >> 0) & 15;
        r->coeffs[8 * i + 3] = (a[4 * i + 1] >> 4) & 15;
        r->coeffs[8 * i + 4] = (a[4 * i + 2] >> 0) & 15;
        r->coeffs[8 * i + 5] = (a[4 * i + 2] >> 4) & 15;
        r->coeffs[8 * i + 6] = (a[4 * i + 3] >> 0) & 15;
        r->coeffs[8 * i + 7] = (a[4 * i + 3] >> 4) & 15;

        r->coeffs[8 * i + 0] = FPTRU_BOUND - r->coeffs[8 * i + 0];
        r->coeffs[8 * i + 1] = FPTRU_BOUND - r->coeffs[8 * i + 1];
        r->coeffs[8 * i + 2] = FPTRU_BOUND - r->coeffs[8 * i + 2];
        r->coeffs[8 * i + 3] = FPTRU_BOUND - r->coeffs[8 * i + 3];
        r->coeffs[8 * i + 4] = FPTRU_BOUND - r->coeffs[8 * i + 4];
        r->coeffs[8 * i + 5] = FPTRU_BOUND - r->coeffs[8 * i + 5];
        r->coeffs[8 * i + 6] = FPTRU_BOUND - r->coeffs[8 * i + 6];
        r->coeffs[8 * i + 7] = FPTRU_BOUND - r->coeffs[8 * i + 7];
    }
    r->coeffs[8 * i + 0] = (a[4 * i + 0] >> 0) & 15;
    r->coeffs[8 * i + 1] = (a[4 * i + 0] >> 4) & 15;
    r->coeffs[8 * i + 2] = (a[4 * i + 1] >> 0) & 15;
    r->coeffs[8 * i + 3] = (a[4 * i + 1] >> 4) & 15;
    r->coeffs[8 * i + 4] = (a[4 * i + 2] >> 0) & 15;
    r->coeffs[8 * i + 0] = FPTRU_BOUND - r->coeffs[8 * i + 0];
    r->coeffs[8 * i + 1] = FPTRU_BOUND - r->coeffs[8 * i + 1];
    r->coeffs[8 * i + 2] = FPTRU_BOUND - r->coeffs[8 * i + 2];
    r->coeffs[8 * i + 3] = FPTRU_BOUND - r->coeffs[8 * i + 3];
    r->coeffs[8 * i + 4] = FPTRU_BOUND - r->coeffs[8 * i + 4];
#elif (FPTRU_N == 761)
    for (i = 0; i < FPTRU_N / 8; ++i)
    {
        r->coeffs[8 * i + 0] = (a[4 * i + 0] >> 0) & 15;
        r->coeffs[8 * i + 1] = (a[4 * i + 0] >> 4) & 15;
        r->coeffs[8 * i + 2] = (a[4 * i + 1] >> 0) & 15;
        r->coeffs[8 * i + 3] = (a[4 * i + 1] >> 4) & 15;
        r->coeffs[8 * i + 4] = (a[4 * i + 2] >> 0) & 15;
        r->coeffs[8 * i + 5] = (a[4 * i + 2] >> 4) & 15;
        r->coeffs[8 * i + 6] = (a[4 * i + 3] >> 0) & 15;
        r->coeffs[8 * i + 7] = (a[4 * i + 3] >> 4) & 15;

        r->coeffs[8 * i + 0] = FPTRU_BOUND - r->coeffs[8 * i + 0];
        r->coeffs[8 * i + 1] = FPTRU_BOUND - r->coeffs[8 * i + 1];
        r->coeffs[8 * i + 2] = FPTRU_BOUND - r->coeffs[8 * i + 2];
        r->coeffs[8 * i + 3] = FPTRU_BOUND - r->coeffs[8 * i + 3];
        r->coeffs[8 * i + 4] = FPTRU_BOUND - r->coeffs[8 * i + 4];
        r->coeffs[8 * i + 5] = FPTRU_BOUND - r->coeffs[8 * i + 5];
        r->coeffs[8 * i + 6] = FPTRU_BOUND - r->coeffs[8 * i + 6];
        r->coeffs[8 * i + 7] = FPTRU_BOUND - r->coeffs[8 * i + 7];
    }
    r->coeffs[8 * i + 0] = (a[4 * i + 0] >> 0) & 15;
    r->coeffs[8 * i + 0] = FPTRU_BOUND - r->coeffs[8 * i + 0];
#endif
}

void pack_ct(unsigned char *r, const poly *a)
{
    unsigned int i;
#if (FPTRU_Q2 == 1024)
    for (i = 0; i < FPTRU_N / 4; ++i)
    {
        r[5 * i + 0] = (a->coeffs[4 * i + 0] >> 0);
        r[5 * i + 1] = (a->coeffs[4 * i + 0] >> 8) | ((int16_t)a->coeffs[4 * i + 1] << 2);
        r[5 * i + 2] = (a->coeffs[4 * i + 1] >> 6) | ((int16_t)a->coeffs[4 * i + 2] << 4);
        r[5 * i + 3] = (a->coeffs[4 * i + 2] >> 4) | ((int16_t)a->coeffs[4 * i + 3] << 6);
        r[5 * i + 4] = (a->coeffs[4 * i + 3] >> 2);
    }
    r[5 * i + 0] = (a->coeffs[4 * i + 0] >> 0);
    r[5 * i + 1] = (a->coeffs[4 * i + 0] >> 8) & 0x3;
#elif (FPTRU_Q2 == 2048)
    for (i = 0; i < FPTRU_N / 8; ++i)
    {
        r[11 * i + 0] = a->coeffs[8 * i + 0] >> 0;
        r[11 * i + 1] = a->coeffs[8 * i + 0] >> 8 | ((int16_t)a->coeffs[8 * i + 1] << 3);
        r[11 * i + 2] = a->coeffs[8 * i + 1] >> 5 | ((int16_t)a->coeffs[8 * i + 2] << 6);
        r[11 * i + 3] = a->coeffs[8 * i + 2] >> 2;
        r[11 * i + 4] = a->coeffs[8 * i + 2] >> 10 | ((int16_t)a->coeffs[8 * i + 3] << 1);
        r[11 * i + 5] = a->coeffs[8 * i + 3] >> 7 | ((int16_t)a->coeffs[8 * i + 4] << 4);
        r[11 * i + 6] = a->coeffs[8 * i + 4] >> 4 | ((int16_t)a->coeffs[8 * i + 5] << 7);
        r[11 * i + 7] = a->coeffs[8 * i + 5] >> 1;
        r[11 * i + 8] = a->coeffs[8 * i + 5] >> 9 | ((int16_t)a->coeffs[8 * i + 6] << 2);
        r[11 * i + 9] = a->coeffs[8 * i + 6] >> 6 | ((int16_t)a->coeffs[8 * i + 7] << 5);
        r[11 * i + 10] = a->coeffs[8 * i + 7] >> 3;
    }
    r[11 * i + 0] = a->coeffs[8 * i + 0] >> 0;
    r[11 * i + 1] = a->coeffs[8 * i + 0] >> 8 | ((int16_t)a->coeffs[8 * i + 1] << 3);
    r[11 * i + 2] = a->coeffs[8 * i + 1] >> 5 | ((int16_t)a->coeffs[8 * i + 2] << 6);
    r[11 * i + 3] = a->coeffs[8 * i + 2] >> 2;
    r[11 * i + 4] = a->coeffs[8 * i + 2] >> 10 | ((int16_t)a->coeffs[8 * i + 3] << 1);
    r[11 * i + 5] = a->coeffs[8 * i + 3] >> 7 | ((int16_t)a->coeffs[8 * i + 4] << 4);
    r[11 * i + 6] = (a->coeffs[8 * i + 4] >> 4) & 0x7f;
#elif (FPTRU_Q2 == 4096)
    for (i = 0; i < FPTRU_N / 2; i++)
    {
        r[3 * i + 0] = (a->coeffs[2 * i] >> 0);
        r[3 * i + 1] = (a->coeffs[2 * i] >> 8) | (a->coeffs[2 * i + 1] << 4);
        r[3 * i + 2] = (a->coeffs[2 * i + 1] >> 4);
    }
    r[3 * i + 0] = (a->coeffs[2 * i] >> 0);
    r[3 * i + 1] = (a->coeffs[2 * i] >> 8) & 0xf;
#endif
}

void unpack_ct(poly *r, const unsigned char *a)
{
    unsigned int i;
#if (FPTRU_Q2 == 1024)
    for (i = 0; i < FPTRU_N / 4; ++i)
    {
        r->coeffs[4 * i + 0] = ((a[5 * i + 0] >> 0) | ((uint16_t)a[5 * i + 1] << 8)) & 0x3FF;
        r->coeffs[4 * i + 1] = ((a[5 * i + 1] >> 2) | ((uint16_t)a[5 * i + 2] << 6)) & 0x3FF;
        r->coeffs[4 * i + 2] = ((a[5 * i + 2] >> 4) | ((uint16_t)a[5 * i + 3] << 4)) & 0x3FF;
        r->coeffs[4 * i + 3] = ((a[5 * i + 3] >> 6) | ((uint16_t)a[5 * i + 4] << 2)) & 0x3FF;
    }
    r->coeffs[4 * i + 0] = ((a[5 * i + 0] >> 0) | ((uint16_t)a[5 * i + 1] << 8)) & 0x3FF;
#elif (FPTRU_Q2 == 2048)
    for (i = 0; i < FPTRU_N / 8; ++i)
    {
        r->coeffs[8 * i + 0] = ((a[11 * i + 0] >> 0) | ((uint16_t)a[11 * i + 1] << 8)) & 0x7FF;
        r->coeffs[8 * i + 1] = ((a[11 * i + 1] >> 3) | ((uint16_t)a[11 * i + 2] << 5)) & 0x7FF;
        r->coeffs[8 * i + 2] = ((a[11 * i + 2] >> 6) | ((uint16_t)a[11 * i + 3] << 2) | ((uint16_t)a[11 * i + 4] << 10)) & 0x7FF;
        r->coeffs[8 * i + 3] = ((a[11 * i + 4] >> 1) | ((uint16_t)a[11 * i + 5] << 7)) & 0x7FF;
        r->coeffs[8 * i + 4] = ((a[11 * i + 5] >> 4) | ((uint16_t)a[11 * i + 6] << 4)) & 0x7FF;
        r->coeffs[8 * i + 5] = ((a[11 * i + 6] >> 7) | ((uint16_t)a[11 * i + 7] << 1) | ((uint16_t)a[11 * i + 8] << 9)) & 0x7FF;
        r->coeffs[8 * i + 6] = ((a[11 * i + 8] >> 2) | ((uint16_t)a[11 * i + 9] << 6)) & 0x7FF;
        r->coeffs[8 * i + 7] = ((a[11 * i + 9] >> 5) | ((uint16_t)a[11 * i + 10] << 3)) & 0x7FF;
    }
    r->coeffs[8 * i + 0] = ((a[11 * i + 0] >> 0) | ((uint16_t)a[11 * i + 1] << 8)) & 0x7FF;
    r->coeffs[8 * i + 1] = ((a[11 * i + 1] >> 3) | ((uint16_t)a[11 * i + 2] << 5)) & 0x7FF;
    r->coeffs[8 * i + 2] = ((a[11 * i + 2] >> 6) | ((uint16_t)a[11 * i + 3] << 2) | ((uint16_t)a[11 * i + 4] << 10)) & 0x7FF;
    r->coeffs[8 * i + 3] = ((a[11 * i + 4] >> 1) | ((uint16_t)a[11 * i + 5] << 7)) & 0x7FF;
    r->coeffs[8 * i + 4] = ((a[11 * i + 5] >> 4) | ((uint16_t)a[11 * i + 6] << 4)) & 0x7FF;
#elif (FPTRU_Q2 == 4096)
    for (i = 0; i < FPTRU_N / 2; i++)
    {
        r->coeffs[2 * i] = ((a[3 * i + 0] >> 0) | ((uint16_t)a[3 * i + 1] << 8)) & 0xFFF;
        r->coeffs[2 * i + 1] = ((a[3 * i + 1] >> 4) | ((uint16_t)a[3 * i + 2] << 4)) & 0xFFF;
    }
    r->coeffs[2 * i] = ((a[3 * i + 0] >> 0) | (((uint16_t)a[3 * i + 1] & 0xF) << 8)) & 0xFFF;
#endif
}

/*公式的验证结果如下
r->coeffs[0] = ((a[11 * idx_h + 0] >> 0) | ((uint16_t)a[11 * idx_h + 1] << 8) | (((uint16_t)a[11 * idx_h + 4] << 10) & 0 )) & 0x7FF
r->coeffs[1] = ((a[11 * idx_h + 1] >> 3) | ((uint16_t)a[11 * idx_h + 2] << 5) | (((uint16_t)a[11 * idx_h + 4] << 10) & 0 )) & 0x7FF
r->coeffs[2] = ((a[11 * idx_h + 2] >> 6) | ((uint16_t)a[11 * idx_h + 3] << 2) | (((uint16_t)a[11 * idx_h + 4] << 10) & 65535 )) & 0x7FF
r->coeffs[3] = ((a[11 * idx_h + 4] >> 1) | ((uint16_t)a[11 * idx_h + 5] << 7) | (((uint16_t)a[11 * idx_h + 8] << 9) & 0 )) & 0x7FF
r->coeffs[4] = ((a[11 * idx_h + 5] >> 4) | ((uint16_t)a[11 * idx_h + 6] << 4) | (((uint16_t)a[11 * idx_h + 8] << 9) & 0 )) & 0x7FF
r->coeffs[5] = ((a[11 * idx_h + 6] >> 7) | ((uint16_t)a[11 * idx_h + 7] << 1) | (((uint16_t)a[11 * idx_h + 8] << 9) & 65535 )) & 0x7FF
r->coeffs[6] = ((a[11 * idx_h + 8] >> 2) | ((uint16_t)a[11 * idx_h + 9] << 6) | (((uint16_t)a[11 * idx_h + 12] << 8) & 0 )) & 0x7FF
r->coeffs[7] = ((a[11 * idx_h + 9] >> 5) | ((uint16_t)a[11 * idx_h + 10] << 3) | (((uint16_t)a[11 * idx_h + 12] << 8) & 0 )) & 0x7FF
存在的问题:对于全局内存的访问占据了太多的时间
*/
__device__ void unpack_ct_cuda(poly * r,const unsigned char *a){
    uint16_t idx_l = threadIdx.x & 7;
    uint16_t idx_h = threadIdx.x >> 3;

    uint8_t l_3_l = idx_l % 3;
    uint8_t l_3_h = idx_l / 3;

    //unsigned char atmp[3];

    /*放到寄存器里没效果
    atmp[0]= a[11 * idx_h + (4 * l_3_h + l_3_l)];
    atmp[1]= a[11 * idx_h + (4 * l_3_h + l_3_l + 1)];
    atmp[2]= a[11 * idx_h + ((l_3_h +1) *4)];
    r->coeffs[threadIdx.x] = ((atmp[0] >> (l_3_h + l_3_l*3)) | ((uint16_t)atmp[1] << (8 - l_3_h - l_3_l*3)) | (((uint16_t)atmp[2] << (10 - l_3_h)) & ((!((idx_l + 1) % 3)) * 0xffff) )) & 0x7FF;*/

    r->coeffs[threadIdx.x]=((a[11 * idx_h + (4 * l_3_h + l_3_l)] >> (l_3_h + l_3_l*3)) | ((uint16_t)a[11 * idx_h + (4 * l_3_h + l_3_l + 1)] << (8 - l_3_h - l_3_l*3)) | (((uint16_t)a[11 * idx_h + ((l_3_h +1) *4)] << (10 - l_3_h)) & ((!((idx_l + 1) % 3)) * 0xffff) )) & 0x7FF;

}
__global__ void unpack_ct_batch(poly * r,const unsigned char *a){
    unpack_ct_cuda(&r[blockIdx.x],&a[blockIdx.x * FPTRU_KEM_CIPHERTEXTBYTES]);
}


/*2024-7-8*/
__device__ void pack_ct_1(unsigned char *r, const poly *a){
    int tid = threadIdx.x;
#if (FPTRU_Q2 == 1024)
    if(tid == FPTRU_N / 4){
        r[5 * tid + 0] = (a->coeffs[4 * tid + 0] >> 0);
        r[5 * tid + 1] = (a->coeffs[4 * tid + 0] >> 8) & 0x3;
    }
    else{
        r[5 * tid + 0] = (a->coeffs[4 * tid + 0] >> 0);
        r[5 * tid + 1] = (a->coeffs[4 * tid + 0] >> 8) | ((int16_t)a->coeffs[4 * tid + 1] << 2);
        r[5 * tid + 2] = (a->coeffs[4 * tid + 1] >> 6) | ((int16_t)a->coeffs[4 * tid + 2] << 4);
        r[5 * tid + 3] = (a->coeffs[4 * tid + 2] >> 4) | ((int16_t)a->coeffs[4 * tid + 3] << 6);
        r[5 * tid + 4] = (a->coeffs[4 * tid + 3] >> 2);
    }
#elif (FPTRU_Q2 == 2048)
    if(tid == FPTRU_N/8){
        r[11 * tid + 0] = a->coeffs[8 * tid + 0] >> 0;
        r[11 * tid + 1] = a->coeffs[8 * tid + 0] >> 8 | ((int16_t)a->coeffs[8 * tid + 1] << 3);
        r[11 * tid + 2] = a->coeffs[8 * tid + 1] >> 5 | ((int16_t)a->coeffs[8 * tid + 2] << 6);
        r[11 * tid + 3] = a->coeffs[8 * tid + 2] >> 2;
        r[11 * tid + 4] = a->coeffs[8 * tid + 2] >> 10 | ((int16_t)a->coeffs[8 * tid + 3] << 1);
        r[11 * tid + 5] = a->coeffs[8 * tid + 3] >> 7 | ((int16_t)a->coeffs[8 * tid + 4] << 4);
        r[11 * tid + 6] = (a->coeffs[8 * tid + 4] >> 4) & 0x7f;
    }
    else{
        r[11 * tid + 0] = a->coeffs[8 * tid + 0] >> 0;
        r[11 * tid + 1] = a->coeffs[8 * tid + 0] >> 8 | ((int16_t)a->coeffs[8 * tid + 1] << 3);
        r[11 * tid + 2] = a->coeffs[8 * tid + 1] >> 5 | ((int16_t)a->coeffs[8 * tid + 2] << 6);
        r[11 * tid + 3] = a->coeffs[8 * tid + 2] >> 2;
        r[11 * tid + 4] = a->coeffs[8 * tid + 2] >> 10 | ((int16_t)a->coeffs[8 * tid + 3] << 1);
        r[11 * tid + 5] = a->coeffs[8 * tid + 3] >> 7 | ((int16_t)a->coeffs[8 * tid + 4] << 4);
        r[11 * tid + 6] = a->coeffs[8 * tid + 4] >> 4 | ((int16_t)a->coeffs[8 * tid + 5] << 7);
        r[11 * tid + 7] = a->coeffs[8 * tid + 5] >> 1;
        r[11 * tid + 8] = a->coeffs[8 * tid + 5] >> 9 | ((int16_t)a->coeffs[8 * tid + 6] << 2);
        r[11 * tid + 9] = a->coeffs[8 * tid + 6] >> 6 | ((int16_t)a->coeffs[8 * tid + 7] << 5);
        r[11 * tid + 10] = a->coeffs[8 * tid + 7] >> 3;
    }

#elif (FPTRU_Q2 == 4096)
    if(tid == FPTRU_N / 2){
        r[3 * tid + 0] = (a->coeffs[2 * tid] >> 0);
        r[3 * tid + 1] = (a->coeffs[2 * tid] >> 8) & 0xf;
    }
    else{
        r[3 * tid + 0] = (a->coeffs[2 * tid] >> 0);
        r[3 * tid + 1] = (a->coeffs[2 * tid] >> 8) | (a->coeffs[2 * tid + 1] << 4);
        r[3 * tid + 2] = (a->coeffs[2 * tid + 1] >> 4);
    }

#endif
}
/*2024-7-8*/
__global__ void pack_ct_batch(unsigned char *r, const poly *a){
    //调试一下

    /*if(threadIdx.x==0 && blockIdx.x ==0){
       printf("in thread look sigma 0\n");
        for(int i=0;i<FPTRU_N;i++){
            printf("%d,",a[blockIdx.x].coeffs[i]);
        }
        printf("\n");


        printf("in thread look sigma 1\n");
        for(int i=0;i<FPTRU_N;i++){
            printf("%d,",a[blockIdx.x + 1].coeffs[i]);
        }
        printf("\n");

        printf("look sigma\n");
        for(int j=0;j<BATCH_SIZE;j++){
            for(int i=0;i<FPTRU_N;i++){
                printf("%d,",a[j].coeffs[i]);
            }
            printf("\n");
        }

    }
    __syncthreads();*/

    pack_ct_1(&r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x],&a[blockIdx.x]);
}


__device__ void Decode_gpu(uint16_t *out, unsigned char *S, uint16_t *M, const long long len)
{


    if (len == 1)
    {
        // if (M[0] == 1)
        //     *out = 0;
        // else if (M[0] <= 256)
        //     *out = uint32_mod_uint14(S[0], M[0]);
        // else
        //     *out = uint32_mod_uint14(S[0] + (((uint16_t)S[1]) << 8), M[0]);
    }

    if (len > 1)
    {
        uint16_t R2[(FPTRU_N + 1) / 2];
        uint16_t M2[(FPTRU_N + 1) / 2];
        uint16_t bottomr[FPTRU_N / 2];
        uint32_t bottomt[FPTRU_N / 2];
        long long i;
        // for (i = 0; i < len - 1; i += 2)
        // {
        //     uint32_t m = M[i] * (uint32_t)M[i + 1];
        //     if (m > 256 * 16383)
        //     {
        //         bottomt[i / 2] = 256 * 256;
        //         bottomr[i / 2] = S[0] + 256 * S[1];
        //         S += 2;
        //         M2[i / 2] = (((m + 255) >> 8) + 255) >> 8;
        //     }
        //     else if (m >= 16384)
        //     {
        //         bottomt[i / 2] = 256;
        //         bottomr[i / 2] = S[0];
        //         S += 1;
        //         M2[i / 2] = (m + 255) >> 8;
        //     }
        //     else
        //     {
        //         bottomt[i / 2] = 1;
        //         bottomr[i / 2] = 0;
        //         M2[i / 2] = m;
        //     }
        // }
        // if (i < len)
        //     M2[i / 2] = M[i];

        Decode_gpu(R2, S, M2, (len + 1) / 2);

        // for (i = 0; i < len - 1; i += 2)
        // {
        //     uint32_t r = bottomr[i / 2];
        //     uint32_t r1;
        //     uint16_t r0;
        //     r += bottomt[i / 2] * R2[i / 2];
        //     uint32_divmod_uint14(&r1, &r0, r, M[i]);
        //     r1 = uint32_mod_uint14(r1, M[i + 1]);
        //     *out++ = r0;
        //     *out++ = r1;
        // }
        // if (i < len)
        //     *out++ = R2[i / 2];
    }
}

__device__ void unpack_pk_gpu(poly *r, unsigned char *a)
{
    uint16_t R[FPTRU_N], M[FPTRU_N];

    for (int i = 0; i < FPTRU_N; ++i)
        M[i] = FPTRU_Q;

    Decode_gpu(R, a, M, FPTRU_N);

    for (int i = 0; i < FPTRU_N; ++i)
        r->coeffs[i] = ((int16_t)R[i]);
}

__global__ void unpack_pk_batch(poly *r, unsigned char *a){
    unpack_pk_gpu(&r[blockIdx.x],&a[blockIdx.x*FPTRU_KEM_PUBLICKEYBYTES]);
}
/*2024-7-28:
线程组织形式
<<<BATCH_SIZE,FPTRU_N/8 + 1(82),0,stream>>>*/
#include "coding.h"
#include "reduce.h"
__global__ void poly_fqcsubq_encode_compress_batch_pack_ct_653(unsigned char *r, poly *array_sigma, unsigned char *arrar_msg, int interval){
    unsigned char * msg = &arrar_msg[interval * blockIdx.x];
    //__shared__ poly sigma;

    //如果读一次很耗时的话，那么不如直接自己读sigma 
    int16_t coes[8];
    coes[0] = array_sigma[blockIdx.x].coeffs[threadIdx.x * 8];
    coes[1] = array_sigma[blockIdx.x].coeffs[threadIdx.x * 8 + 1];
    coes[2] = array_sigma[blockIdx.x].coeffs[threadIdx.x * 8 + 2];
    coes[3] = array_sigma[blockIdx.x].coeffs[threadIdx.x * 8 + 3];
    coes[4] = array_sigma[blockIdx.x].coeffs[threadIdx.x * 8 + 4];

    //TODO:设计一下，使得idx为FPTRU_N/8时,读取的值少一点->1.测试一下和使用分支的区别 2.能否形成合并内存访问，会不会造成浪费
    coes[5] = array_sigma[blockIdx.x].coeffs[(threadIdx.x * 8 + 5) * (threadIdx.x != FPTRU_N/8)];
    coes[6] = array_sigma[blockIdx.x].coeffs[(threadIdx.x * 8 + 6) * (threadIdx.x != FPTRU_N/8)];
    coes[7] = array_sigma[blockIdx.x].coeffs[(threadIdx.x * 8 + 7) * (threadIdx.x != FPTRU_N/8)];

    int k = threadIdx.x >> 1;

    uint8_t tmp,mh;
    int16_t mask,t;

    tmp = (threadIdx.x >= FPTRU_MSGBYTES * 2) ? 0 : (msg[k] >> ((threadIdx.x %2 ) << 2)) & 0xF; //这里的判断语句和一般的判断语句有什么区别呢
    /*if(threadIdx.x >= FPTRU_MSGBYTES * 2){
        tmp = 0;
    }
    else{
        tmp = (msg[k] >> ((threadIdx.x %2 ) << 2)) & 0xF;
    }*/
    mh = encode_e8(tmp);

    //后面可以一直使用mh

    mask = -(int16_t)((mh >> 0) & 1);
    t = (int32_t)((fqcsubq(coes[0]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
    t = t + (mask & (FPTRU_Q2 >> 1));
    coes[0] = t & (FPTRU_Q2 - 1);

    mask = -(int16_t)((mh >> 1) & 1);
    t = (int32_t)((fqcsubq(coes[1]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
    t = t + (mask & (FPTRU_Q2 >> 1));
    coes[1] = t & (FPTRU_Q2 - 1);

    mask = -(int16_t)((mh >> 2) & 1);
    t = (int32_t)((fqcsubq(coes[2]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
    t = t + (mask & (FPTRU_Q2 >> 1));
    coes[2] = t & (FPTRU_Q2 - 1);

    mask = -(int16_t)((mh >> 3) & 1);
    t = (int32_t)((fqcsubq(coes[3]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
    t = t + (mask & (FPTRU_Q2 >> 1));
    coes[3] = t & (FPTRU_Q2 - 1);

    mask = -(int16_t)((mh >> 4) & 1);
    t = (int32_t)((fqcsubq(coes[4]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
    t = t + (mask & (FPTRU_Q2 >> 1));
    coes[4] = t & (FPTRU_Q2 - 1);

    mask = -(int16_t)((mh >> 5) & 1);
    t = (int32_t)((fqcsubq(coes[5]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
    t = t + (mask & (FPTRU_Q2 >> 1));
    coes[5] = t & (FPTRU_Q2 - 1);

    mask = -(int16_t)((mh >> 6) & 1);
    t = (int32_t)((fqcsubq(coes[6]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
    t = t + (mask & (FPTRU_Q2 >> 1));
    coes[6] = t & (FPTRU_Q2 - 1);

    mask = -(int16_t)((mh >> 7) & 1);
    t = (int32_t)((fqcsubq(coes[7]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
    t = t + (mask & (FPTRU_Q2 >> 1));
    coes[7] = t & (FPTRU_Q2 - 1);

    r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 11 * threadIdx.x + 0] = coes[0] >> 0;
    r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 11 * threadIdx.x + 1] = coes[0] >> 8 | ((int16_t)coes[1] << 3);
    r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 11 * threadIdx.x + 2] = coes[1] >> 5 | ((int16_t)coes[2] << 6);
    r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 11 * threadIdx.x + 3] = coes[2] >> 2;
    r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 11 * threadIdx.x + 4] = coes[2] >> 10 | ((int16_t)coes[3] << 1);
    r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 11 * threadIdx.x + 5] = coes[3] >> 7 | ((int16_t)coes[4] << 4);
    if(threadIdx.x < FPTRU_N/8){
        //做额外的赋值操作
        r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 11 * threadIdx.x + 6] = coes[4] >> 4 | ((int16_t)coes[5] << 7);
        r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 11 * threadIdx.x + 7] = coes[5] >> 1;
        r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 11 * threadIdx.x + 8] = coes[5] >> 9 | ((int16_t)coes[6] << 2);
        r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 11 * threadIdx.x + 9] = coes[6] >> 6 | ((int16_t)coes[7] << 5);
        r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 11 * threadIdx.x + 10] = coes[7] >> 3;
    }
    else{
        r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 11 * threadIdx.x + 6] = (coes[4] >> 4) & 0x7f;
    }
}


__global__ void poly_fqcsubq_encode_compress_batch_pack_ct_761(unsigned char *r, poly *array_sigma, unsigned char *arrar_msg, int interval){
    unsigned char * msg = &arrar_msg[interval * blockIdx.x];
    //__shared__ poly sigma;

    //如果读一次很耗时的话，那么不如直接自己读sigma 
    int16_t coes[8];
    coes[0] = array_sigma[blockIdx.x].coeffs[threadIdx.x * 8];
    coes[1] = array_sigma[blockIdx.x].coeffs[(threadIdx.x * 8 + 1) * (threadIdx.x != FPTRU_N/8)];
    coes[2] = array_sigma[blockIdx.x].coeffs[(threadIdx.x * 8 + 2) * (threadIdx.x != FPTRU_N/8)];
    coes[3] = array_sigma[blockIdx.x].coeffs[(threadIdx.x * 8 + 3) * (threadIdx.x != FPTRU_N/8)];
    coes[4] = array_sigma[blockIdx.x].coeffs[(threadIdx.x * 8 + 4) * (threadIdx.x != FPTRU_N/8)];
    coes[5] = array_sigma[blockIdx.x].coeffs[(threadIdx.x * 8 + 5) * (threadIdx.x != FPTRU_N/8)];
    coes[6] = array_sigma[blockIdx.x].coeffs[(threadIdx.x * 8 + 6) * (threadIdx.x != FPTRU_N/8)];
    coes[7] = array_sigma[blockIdx.x].coeffs[(threadIdx.x * 8 + 7) * (threadIdx.x != FPTRU_N/8)];

    int k = threadIdx.x >> 1;

    uint8_t tmp,mh;
    int16_t mask,t;

    tmp = (threadIdx.x >= FPTRU_MSGBYTES * 2) ? 0 : (msg[k] >> ((threadIdx.x %2 ) << 2)) & 0xF; //这里的判断语句和一般的判断语句有什么区别呢
    /*if(threadIdx.x >= FPTRU_MSGBYTES * 2){
        tmp = 0;
    }
    else{
        tmp = (msg[k] >> ((threadIdx.x %2 ) << 2)) & 0xF;
    }*/
    mh = encode_e8(tmp);

    //后面可以一直使用mh

    mask = -(int16_t)((mh >> 0) & 1);
    t = (int32_t)((fqcsubq(coes[0]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
    t = t + (mask & (FPTRU_Q2 >> 1));
    coes[0] = t & (FPTRU_Q2 - 1);

    mask = -(int16_t)((mh >> 1) & 1);
    t = (int32_t)((fqcsubq(coes[1]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
    t = t + (mask & (FPTRU_Q2 >> 1));
    coes[1] = t & (FPTRU_Q2 - 1);

    mask = -(int16_t)((mh >> 2) & 1);
    t = (int32_t)((fqcsubq(coes[2]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
    t = t + (mask & (FPTRU_Q2 >> 1));
    coes[2] = t & (FPTRU_Q2 - 1);

    mask = -(int16_t)((mh >> 3) & 1);
    t = (int32_t)((fqcsubq(coes[3]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
    t = t + (mask & (FPTRU_Q2 >> 1));
    coes[3] = t & (FPTRU_Q2 - 1);

    mask = -(int16_t)((mh >> 4) & 1);
    t = (int32_t)((fqcsubq(coes[4]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
    t = t + (mask & (FPTRU_Q2 >> 1));
    coes[4] = t & (FPTRU_Q2 - 1);

    mask = -(int16_t)((mh >> 5) & 1);
    t = (int32_t)((fqcsubq(coes[5]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
    t = t + (mask & (FPTRU_Q2 >> 1));
    coes[5] = t & (FPTRU_Q2 - 1);

    mask = -(int16_t)((mh >> 6) & 1);
    t = (int32_t)((fqcsubq(coes[6]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
    t = t + (mask & (FPTRU_Q2 >> 1));
    coes[6] = t & (FPTRU_Q2 - 1);

    mask = -(int16_t)((mh >> 7) & 1);
    t = (int32_t)((fqcsubq(coes[7]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
    t = t + (mask & (FPTRU_Q2 >> 1));
    coes[7] = t & (FPTRU_Q2 - 1);

    
    if(threadIdx.x == FPTRU_N/8){//95
        //做额外的赋值操作
        r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 5 * (threadIdx.x * 2) + 0] = coes[0] >> 0;
        r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 5 * (threadIdx.x * 2) + 1] = (coes[0] >> 8) & 0x3;
    }
    else{//0-94
        r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 5 * (threadIdx.x * 2) + 0] = coes[0] >> 0;
        r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 5 * (threadIdx.x * 2 + 1) + 0] = coes[4] >> 0;

        r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 5 * (threadIdx.x * 2) + 1] = (coes[0] >> 8) | ((int16_t)coes[1] << 2);
        r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 5 * (threadIdx.x * 2 + 1) + 1] = (coes[4] >> 8) | ((int16_t)coes[5] << 2);

        r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 5 * (threadIdx.x * 2) + 2] = (coes[1] >> 6) | ((int16_t)coes[2] << 4);
        r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 5 * (threadIdx.x * 2 + 1) + 2] = (coes[5] >> 6) | ((int16_t)coes[6] << 4);

        r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 5 * (threadIdx.x * 2) + 3] = (coes[2] >> 4) | ((int16_t)coes[3] << 6);
        r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 5 * (threadIdx.x * 2 + 1) + 3] = (coes[6] >> 4) | ((int16_t)coes[7] << 6);

        r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 5 * (threadIdx.x * 2) + 4] = coes[3] >> 2;;
        r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 5 * (threadIdx.x * 2 + 1) + 4] = coes[7] >> 2;
    }
}


__global__ void poly_fqcsubq_encode_compress_batch_pack_ct_1277(unsigned char *r, poly *array_sigma, unsigned char *arrar_msg, int interval){
    unsigned char * msg = &arrar_msg[interval * blockIdx.x];
    //__shared__ poly sigma;

    //如果读一次很耗时的话，那么不如直接自己读sigma 
    int16_t coes[8];
    coes[0] = array_sigma[blockIdx.x].coeffs[threadIdx.x * 8];
    coes[1] = array_sigma[blockIdx.x].coeffs[threadIdx.x * 8 + 1];
    coes[2] = array_sigma[blockIdx.x].coeffs[threadIdx.x * 8 + 2];
    coes[3] = array_sigma[blockIdx.x].coeffs[threadIdx.x * 8 + 3];
    coes[4] = array_sigma[blockIdx.x].coeffs[threadIdx.x * 8 + 4];
    coes[5] = array_sigma[blockIdx.x].coeffs[(threadIdx.x * 8 + 5) * (threadIdx.x != FPTRU_N/8)];
    coes[6] = array_sigma[blockIdx.x].coeffs[(threadIdx.x * 8 + 6) * (threadIdx.x != FPTRU_N/8)];
    coes[7] = array_sigma[blockIdx.x].coeffs[(threadIdx.x * 8 + 7) * (threadIdx.x != FPTRU_N/8)];

    int k = threadIdx.x >> 1;

    uint8_t tmp,mh;
    int16_t mask,t;

    tmp = (threadIdx.x >= FPTRU_MSGBYTES * 2) ? 0 : (msg[k] >> ((threadIdx.x %2 ) << 2)) & 0xF; //这里的判断语句和一般的判断语句有什么区别呢
    /*if(threadIdx.x >= FPTRU_MSGBYTES * 2){
        tmp = 0;
    }
    else{
        tmp = (msg[k] >> ((threadIdx.x %2 ) << 2)) & 0xF;
    }*/
    mh = encode_e8(tmp);

    //后面可以一直使用mh

    mask = -(int16_t)((mh >> 0) & 1);
    t = (int32_t)((fqcsubq(coes[0]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
    t = t + (mask & (FPTRU_Q2 >> 1));
    coes[0] = t & (FPTRU_Q2 - 1);

    mask = -(int16_t)((mh >> 1) & 1);
    t = (int32_t)((fqcsubq(coes[1]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
    t = t + (mask & (FPTRU_Q2 >> 1));
    coes[1] = t & (FPTRU_Q2 - 1);

    mask = -(int16_t)((mh >> 2) & 1);
    t = (int32_t)((fqcsubq(coes[2]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
    t = t + (mask & (FPTRU_Q2 >> 1));
    coes[2] = t & (FPTRU_Q2 - 1);

    mask = -(int16_t)((mh >> 3) & 1);
    t = (int32_t)((fqcsubq(coes[3]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
    t = t + (mask & (FPTRU_Q2 >> 1));
    coes[3] = t & (FPTRU_Q2 - 1);

    mask = -(int16_t)((mh >> 4) & 1);
    t = (int32_t)((fqcsubq(coes[4]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
    t = t + (mask & (FPTRU_Q2 >> 1));
    coes[4] = t & (FPTRU_Q2 - 1);

    mask = -(int16_t)((mh >> 5) & 1);
    t = (int32_t)((fqcsubq(coes[5]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
    t = t + (mask & (FPTRU_Q2 >> 1));
    coes[5] = t & (FPTRU_Q2 - 1);

    mask = -(int16_t)((mh >> 6) & 1);
    t = (int32_t)((fqcsubq(coes[6]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
    t = t + (mask & (FPTRU_Q2 >> 1));
    coes[6] = t & (FPTRU_Q2 - 1);

    mask = -(int16_t)((mh >> 7) & 1);
    t = (int32_t)((fqcsubq(coes[7]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
    t = t + (mask & (FPTRU_Q2 >> 1));
    coes[7] = t & (FPTRU_Q2 - 1);

    r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 5 * (threadIdx.x * 2) + 0] = coes[0] >> 0;
    r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 5 * (threadIdx.x * 2) + 1] = (coes[0] >> 8) | ((int16_t)coes[1] << 2);
    r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 5 * (threadIdx.x * 2) + 2] = (coes[1] >> 6) | ((int16_t)coes[2] << 4);
    r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 5 * (threadIdx.x * 2) + 3] = (coes[2] >> 4) | ((int16_t)coes[3] << 6);
    r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 5 * (threadIdx.x * 2) + 4] = coes[3] >> 2;;
    
    if(threadIdx.x == FPTRU_N/8){ //159
        //做额外的赋值操作
        r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 5 * (threadIdx.x * 2 + 1) + 0] = coes[4] >> 0;
        r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 5 * (threadIdx.x * 2 + 1) + 1] = (coes[4] >> 8) & 0x3;
    }
    else{ //0-158
        
        r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 5 * (threadIdx.x * 2 + 1) + 0] = coes[4] >> 0;

        
        r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 5 * (threadIdx.x * 2 + 1) + 1] = (coes[4] >> 8) | ((int16_t)coes[5] << 2);

        
        r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 5 * (threadIdx.x * 2 + 1) + 2] = (coes[5] >> 6) | ((int16_t)coes[6] << 4);

        
        r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 5 * (threadIdx.x * 2 + 1) + 3] = (coes[6] >> 4) | ((int16_t)coes[7] << 6);

        
        r[FPTRU_PKE_CIPHERTEXTBYTES*blockIdx.x + 5 * (threadIdx.x * 2 + 1) + 4] = coes[7] >> 2;
    }
}