#ifndef PARAMS_H
#define PARAMS_H

#ifndef FPTRU_N

#define FPTRU_N 1277

#endif

#define USING_PK_ENCODE 1

#ifndef BATCH_SIZE
#define BATCH_SIZE 1000
#endif

#ifndef NUM_THREAD
#define NUM_THREAD 10
#endif

#define HXWTEST 100

//#define dg

/*2024-3-14:
宏对应的实际含义:FPTRU_Q:素阶数域中的元素的模数
                FPTRU_Q2:密文的模数

*/
#if (FPTRU_N == 653)
#define FPTRU_Q 4621
#define FPTRU_LOGQ 13 //代表的是FPTRU_Q的比特数
#define FPTRU_Q2 2048
#define FPTRU_LOGQ2 11
#define FPTRU_BOUND 7
#define FPTRU_COIN_BYTES (((FPTRU_N * 6 + 7) / 8) * 2)//653使用cbd3,每一个系数需要6bit,则共需要FPTRU_N * 6 bit，再将其转化为以字节为单位，由于一次需要生成两个多项式，因此再乘上2

#elif (FPTRU_N == 761)
#define FPTRU_Q 4591
#define FPTRU_LOGQ 13
#define FPTRU_Q2 1024
#define FPTRU_LOGQ2 10
#define FPTRU_BOUND 5
#define FPTRU_COIN_BYTES (((FPTRU_N * 4 + 7) / 8) * 2)//761使用cbd2,每一个系数需要4bit,则共需要FPTRU_N * 4 bit，再将其转化为以字节为单位，由于一次需要生成两个多项式，因此再乘上2

#elif (FPTRU_N == 1277)
#define FPTRU_Q 7879
#define FPTRU_LOGQ 13
#define FPTRU_Q2 1024
#define FPTRU_LOGQ2 10
#define FPTRU_BOUND 5
#define FPTRU_COIN_BYTES (((FPTRU_N * 4 + 7) / 8) * 2)//1277使用cbd2,每一个系数需要4bit,则共需要FPTRU_N * 4 bit，再将其转化为以字节为单位，由于一次需要生成两个多项式，因此再乘上2
#endif

//
#define FPTRU_SEEDBYTES 32
#define FPTRU_SHAREDKEYBYTES 32
#define FPTRU_MSGBYTES (FPTRU_N / 16)

#if (USING_PK_ENCODE == 1)
#define FPTRU_Q_HALF ((FPTRU_Q - 1) / 2)
#if (FPTRU_N == 653)
#define FPTRU_PKE_PUBLICKEYBYTES 994
#elif (FPTRU_N == 761)
#define FPTRU_PKE_PUBLICKEYBYTES 1158
#elif (FPTRU_N == 1277)
#define FPTRU_PKE_PUBLICKEYBYTES 2067
#endif
#endif

#ifndef dg
#define FPTRU_PKE_PUBLICKEYBYTES ((FPTRU_LOGQ * FPTRU_N + 7) / 8)
#endif

#define FPTRU_PKE_CIPHERTEXTBYTES ((FPTRU_LOGQ2 * FPTRU_N + 7) / 8) //3-14:密文长度
#define FPTRU_PKE_SECRETKEYBYTES ((4 * FPTRU_N + 7) / 8) //3-14:TODO:需要确认私钥是如何产生的，进一步确认为什么私钥中的每一个元素只需要4bit

#define FPTRU_KEM_PUBLICKEYBYTES FPTRU_PKE_PUBLICKEYBYTES
#define FPTRU_KEM_SECRETKEYBYTES (FPTRU_PKE_SECRETKEYBYTES + FPTRU_PKE_PUBLICKEYBYTES + FPTRU_SEEDBYTES)//TODO:SEEDBYTES是对应了keygen中的z吗？
#define FPTRU_KEM_CIPHERTEXTBYTES FPTRU_PKE_CIPHERTEXTBYTES
#define FPTRU_PREFIXHASHBYTES 33

#endif
