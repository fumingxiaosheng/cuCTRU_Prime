#ifndef KEM_H
#define KEM_H

float crypto_kem_keygen(unsigned char *array_pk,unsigned char *array_sk,cudaStream_t stream);
int crypto_kem_keygen_single(unsigned char *array_pk,unsigned char *array_sk);

int crypto_kem_keygen_single_v2(unsigned char *array_pk,unsigned char *array_sk);

void crypto_kem_keygen_v2(unsigned char *array_pk,unsigned char *array_sk,cudaStream_t stream);

double crypto_kem_keygen_no_inv_batch(unsigned char *array_pk,unsigned char *array_sk,cudaStream_t stream);

void crypto_kem_keygen_tensor_core(unsigned char *array_pk,unsigned char *array_sk,cudaStream_t stream);

double crypto_kem_encaps(unsigned char *ct, unsigned char *k, const unsigned char *pk,cudaStream_t stream);
double crypto_kem_encaps_stream_order(unsigned char *ct, unsigned char *k, const unsigned char *pk,cudaStream_t stream);
double crypto_kem_keygen_no_inv_batch_stream_ordered_async(unsigned char *array_pk,unsigned char *array_sk,cudaStream_t stream);

void crypto_kem_decaps(unsigned char *k, const unsigned char *ct, const unsigned char *sk,cudaStream_t stream);

void crypto_kem_decaps_v2(unsigned char *k, const unsigned char *ct, const unsigned char *sk,cudaStream_t stream);

double crypto_kem_decaps_v3(unsigned char *k, const unsigned char *ct, const unsigned char *sk,cudaStream_t stream, int *res);

double crypto_kem_encaps_v4(unsigned char *ct, unsigned char *k, const unsigned char *pk,cudaStream_t stream);

double fptru_keygen(unsigned char *array_pk,unsigned char *array_sk);

void fptru_encaps(unsigned char *ct, unsigned char *k,  unsigned char *pk);

void fptru_decaps(unsigned char *k, unsigned char *ct, unsigned char *sk, int *res, unsigned char *k1);
#endif