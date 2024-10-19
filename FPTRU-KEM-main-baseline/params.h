#ifndef PARAMS_H
#define PARAMS_H

#ifndef FPTRU_N

#define FPTRU_N 761

#endif

#define USING_PK_ENCODE 1

#if (FPTRU_N == 653)
#define FPTRU_Q 4621
#define FPTRU_LOGQ 13
#define FPTRU_Q2 2048
#define FPTRU_LOGQ2 11
#define FPTRU_BOUND 7
#define FPTRU_COIN_BYTES (((FPTRU_N * 6 + 7) / 8) * 2)

#elif (FPTRU_N == 761)
#define FPTRU_Q 4591
#define FPTRU_LOGQ 13
#define FPTRU_Q2 1024
#define FPTRU_LOGQ2 10
#define FPTRU_BOUND 5
#define FPTRU_COIN_BYTES (((FPTRU_N * 4 + 7) / 8) * 2)

#elif (FPTRU_N == 1277)
#define FPTRU_Q 7879
#define FPTRU_LOGQ 13
#define FPTRU_Q2 1024
#define FPTRU_LOGQ2 10
#define FPTRU_BOUND 5
#define FPTRU_COIN_BYTES (((FPTRU_N * 4 + 7) / 8) * 2)
#endif

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
#elif (USING_PK_ENCODE == 0)
#define FPTRU_PKE_PUBLICKEYBYTES ((FPTRU_LOGQ * FPTRU_N + 7) / 8)
#endif

#define FPTRU_PKE_CIPHERTEXTBYTES ((FPTRU_LOGQ2 * FPTRU_N + 7) / 8)
#define FPTRU_PKE_SECRETKEYBYTES ((4 * FPTRU_N + 7) / 8)

#define FPTRU_KEM_PUBLICKEYBYTES FPTRU_PKE_PUBLICKEYBYTES
#define FPTRU_KEM_SECRETKEYBYTES (FPTRU_PKE_SECRETKEYBYTES + FPTRU_PKE_PUBLICKEYBYTES + FPTRU_SEEDBYTES)
#define FPTRU_KEM_CIPHERTEXTBYTES FPTRU_PKE_CIPHERTEXTBYTES
#define FPTRU_PREFIXHASHBYTES 33



#endif
