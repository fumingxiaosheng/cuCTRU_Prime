#include <stddef.h>
#include "randombytes.h"
#include "params.h"
#include "pke.h"
#include "poly.h"
#include "pack.h"
#include "symmetric_crypto.h"
#include "speed.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <utility>
#include <vector>
#include <numeric>

class ChronoTimer {
public:
    explicit ChronoTimer(std::string func_name) {
        func_name_ = std::move(func_name);
    }

    ~ChronoTimer() {
        auto n_trials = time_.size();
        auto mean_time = mean(time_);
        auto median_time = median(time_);
        auto min_time = min(time_);
        auto stddev = std_dev(time_);
        std::cout << func_name_ << ","
                  << n_trials << ","
                  << min_time << ","
                  << median_time << ","
                  << stddev << std::endl;
    }

    inline void start() {
        start_point_ = std::chrono::steady_clock::now();
    }

    inline void stop() {
        stop_point_ = std::chrono::steady_clock::now();
        std::chrono::duration<float, std::micro> elapsed_time = stop_point_ - start_point_;
        time_.emplace_back(elapsed_time.count());
    }

private:
    std::string func_name_;

    std::chrono::time_point<std::chrono::steady_clock> start_point_, stop_point_;
    std::vector<float> time_;

    static float mean(std::vector<float> const &v) {
        if (v.empty())
            return 0;

        auto const count = static_cast<float>(v.size());
        return std::accumulate(v.begin(), v.end(), 0.0f) / count;
    }

    static float median(std::vector<float> v) {
        size_t size = v.size();

        if (size == 0)
            return 0;
        else {
            sort(v.begin(), v.end());
            if (size % 2 == 0)
                return (v[size / 2 - 1] + v[size / 2]) / 2;
            else
                return v[size / 2];
        }
    }

    static float min(std::vector<float> v) {
        size_t size = v.size();

        if (size == 0)
            return 0;

        sort(v.begin(), v.end());
        return v.front();
    }

    static float max(std::vector<float> v) {
        size_t size = v.size();

        if (size == 0)
            return 0;

        sort(v.begin(), v.end());
        return v.back();
    }

    static double std_dev(std::vector<float> const &v) {
        if (v.empty())
            return 0;

        auto const count = static_cast<float>(v.size());
        float mean = std::accumulate(v.begin(), v.end(), 0.0f) / count;

        std::vector<double> diff(v.size());

        std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
        double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        return (sq_sum / count);//remov sqrt
    }
};

ChronoTimer shake256_1("shake256_1");
ChronoTimer sha3_512_1("sha3_512_1");
ChronoTimer shake256_2("shake256_2");

ChronoTimer keygen("keygen");
ChronoTimer encaps("encaps");
ChronoTimer decaps("decpas");
int crypto_kem_keygen(unsigned char *pk,
                      unsigned char *sk)
{
  keygen.start();
  unsigned int i;
  unsigned char coins[FPTRU_COIN_BYTES];

  randombytes(coins, FPTRU_SEEDBYTES);
  crypto_hash_shake256(coins, FPTRU_COIN_BYTES, coins, FPTRU_SEEDBYTES);
  crypto_pke_keygen(pk, sk, coins);

  for (i = 0; i < FPTRU_PKE_PUBLICKEYBYTES; ++i)
    sk[i + FPTRU_PKE_SECRETKEYBYTES] = pk[i];

  randombytes(sk + FPTRU_PKE_SECRETKEYBYTES + FPTRU_PKE_PUBLICKEYBYTES, FPTRU_SEEDBYTES);
  keygen.stop();
  return 0;
}

unsigned char seed[FPTRU_SEEDBYTES]={0x3f,0x9,0x39,0x39,0x1f,0xbb,0x3d,0x43,0xe6,0x73,0x8f,0x5c,0x85,0xfa,0xf6,0x5d,0xc,0xa8,0xf2,0xe2,0x84,0xc8,0xaa,0x73,0xa5,0x5b,0xc1,0xe2,0xed,0xc,0x85,0x16};
int crypto_kem_encaps(unsigned char *ct,
                      unsigned char *k,
                      const unsigned char *pk)
{
  encaps.start();
  unsigned int i;
  unsigned char buf[FPTRU_SHAREDKEYBYTES + FPTRU_COIN_BYTES / 2], m[FPTRU_PREFIXHASHBYTES + FPTRU_MSGBYTES];

  //TODO:在不固定值的情况下，需要打开randombytes
  randombytes(buf, FPTRU_SEEDBYTES);

  for(int k=0;k<FPTRU_SEEDBYTES;k++){
      //printf("0x%x,",buf_h[k]);
      buf[k]=seed[k];
  }
  shake256_1.start();
  crypto_hash_shake256(m + FPTRU_PREFIXHASHBYTES, FPTRU_MSGBYTES, buf, 32);
  shake256_1.stop();

  for (i = 0; i < FPTRU_PREFIXHASHBYTES; ++i)
    m[i] = pk[i];

  sha3_512_1.start();
  crypto_hash_sha3_512(buf, m, FPTRU_PREFIXHASHBYTES + FPTRU_MSGBYTES);//(𝐾, 𝑐𝑜𝑖𝑛) ≔ ℋ(𝐼𝐷(𝑝𝑘), 𝑚)
  sha3_512_1.stop();
  

  shake256_2.start();
  crypto_hash_shake256(buf + FPTRU_SHAREDKEYBYTES, FPTRU_COIN_BYTES / 2, buf + FPTRU_SHAREDKEYBYTES, 32);
  shake256_2.stop();

  crypto_pke_enc(ct, pk, m + FPTRU_PREFIXHASHBYTES, buf + FPTRU_SHAREDKEYBYTES);

  for (i = 0; i < FPTRU_SHAREDKEYBYTES; ++i)
    k[i] = buf[i];
  encaps.stop();
  return 0;
}

int crypto_kem_decaps(unsigned char *k,
                      const unsigned char *ct,
                      const unsigned char *sk)
{
  decaps.start();
  unsigned int i;
  unsigned char buf[FPTRU_SHAREDKEYBYTES + FPTRU_COIN_BYTES / 2], buf2[FPTRU_SHAREDKEYBYTES * 2], m[FPTRU_PREFIXHASHBYTES + FPTRU_MSGBYTES];
  unsigned char ct2[FPTRU_PKE_CIPHERTEXTBYTES + FPTRU_SEEDBYTES + FPTRU_PREFIXHASHBYTES];
  int16_t t;
  int32_t fail;

  crypto_pke_dec(m + FPTRU_PREFIXHASHBYTES, ct, sk);

  for (i = 0; i < FPTRU_PREFIXHASHBYTES; ++i)
    m[i] = sk[i + FPTRU_PKE_SECRETKEYBYTES];

  crypto_hash_sha3_512(buf, m, FPTRU_PREFIXHASHBYTES + FPTRU_MSGBYTES);
  crypto_hash_shake256(buf + FPTRU_SHAREDKEYBYTES, FPTRU_COIN_BYTES / 2, buf + FPTRU_SHAREDKEYBYTES, 32);
  crypto_pke_enc(ct2, sk + FPTRU_PKE_SECRETKEYBYTES, m + FPTRU_PREFIXHASHBYTES, buf + FPTRU_SHAREDKEYBYTES);

  t = 0;
  for (i = 0; i < FPTRU_PKE_CIPHERTEXTBYTES; ++i)
    t |= ct[i] ^ ct2[i];

  fail = (uint16_t)t;
  fail = (-fail) >> 31;

  for (i = 0; i < FPTRU_PREFIXHASHBYTES; ++i)
    ct2[i] = sk[i + FPTRU_PKE_SECRETKEYBYTES];
  for (i = 0; i < FPTRU_SEEDBYTES; ++i)
    ct2[i + FPTRU_PREFIXHASHBYTES] = sk[i + FPTRU_PKE_SECRETKEYBYTES + FPTRU_PKE_PUBLICKEYBYTES];
  for (i = 0; i < FPTRU_PKE_CIPHERTEXTBYTES; ++i)
    ct2[i + FPTRU_PREFIXHASHBYTES + FPTRU_SEEDBYTES] = ct[i];
  crypto_hash_sha3_512(buf2, ct2, FPTRU_PKE_CIPHERTEXTBYTES + FPTRU_SEEDBYTES + FPTRU_PREFIXHASHBYTES);
  for (i = 0; i < FPTRU_SHAREDKEYBYTES; ++i)
    k[i] = buf[i] ^ ((-fail) & (buf[i] ^ buf2[i]));

  decaps.stop();
  return fail;
}
