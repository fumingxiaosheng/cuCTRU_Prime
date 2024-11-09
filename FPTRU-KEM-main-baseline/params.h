#ifndef PARAMS_H
#define PARAMS_H

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

#ifndef FPTRU_N

#define FPTRU_N 761

#endif

#define USING_PK_ENCODE 0

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

#endif
