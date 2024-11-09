#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>
#include "params.h"
#include <stdint.h>


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

#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

/*TODO:为什么把这里删除了就可以成功的编译呢？
#if (FPTRU_N == 653)
#include "./poly_mul_n653q/n653.h" //2024-4-22:在这里添加上了对于n653.h的头文件包括
#endif
*/


typedef struct
{
  int16_t coeffs[FPTRU_N];//使用16位有符号数来表示每一个系数
} poly;

//全局的宏定义应该放在全局
#if (FPTRU_N == 653)
#define poly_mul_q1 poly_radix_ntt_n653_q1
#define poly_mul_q2 poly_radix_ntt_n653_q2

#define poly_mul_q1_batch poly_mul_653_batch_q1//DONE
#define poly_mul_q2_batch poly_radix_ntt_n653_q2
#elif (FPTRU_N == 761)
#define poly_mul_q1 poly_radix_ntt_n761_q1
#define poly_mul_q2 poly_radix_ntt_n761_q2

#define poly_mul_q1_batch poly_radix_ntt_n761_q1
#define poly_mul_q2_batch poly_radix_ntt_n761_q2
#elif (FPTRU_N == 1277)
#define poly_mul_q1 poly_radix_ntt_n1277_q1
#define poly_mul_q2 poly_radix_ntt_n1277_q2

#define poly_mul_q1_batch poly_radix_ntt_n1277_q1
#define poly_mul_q2_batch poly_radix_ntt_n1277_q2
#endif


void HandleError(cudaError_t err, const char* file, int line);
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define BEFORE_SPEED {cudaEventCreate( &start );cudaEventCreate( &stop ) ;cudaEventRecord( start, 0 ) ;}

#define AFTER_SPEED(x) {cudaEventRecord( stop, 0 ) ;cudaEventSynchronize( stop );float elapsedTime; cudaEventElapsedTime( &elapsedTime,start, stop ); printf( "%s: %f us\n",x,elapsedTime*1000 );cudaEventDestroy( start );cudaEventDestroy( stop );}


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
        return std::sqrt(sq_sum / count);
    }
};

class CUDATimer {
public:
    explicit CUDATimer(std::string func_name) {
        func_name_ = std::move(func_name);

        cudaEventCreate(&start_event_);
        cudaEventCreate(&stop_event_);
    }

    ~CUDATimer() {
        cudaEventDestroy(start_event_);
        cudaEventDestroy(stop_event_);
        auto n_trials = time_.size();
        auto mean_time = mean(time_);
        auto min_time = min(time_);
        auto median_time = median(time_);
        auto stddev = std_dev(time_);
        std::cout << func_name_ << ","
                  << n_trials << ","
                  << min_time << ","
                  << median_time << ","
                  << stddev << std::endl;
    }

    inline void start() const {
        cudaEventRecord(start_event_);

    }

    inline void stop() {
        cudaEventRecord(stop_event_);
        cudaEventSynchronize(stop_event_);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_event_, stop_event_);
        time_.push_back(milliseconds);
    }

private:
    std::string func_name_;
    cudaEvent_t start_event_{}, stop_event_{};
    std::vector<float> time_;

    static float mean(std::vector<float> const &v) {
        if (v.empty())
            return 0;

        auto const count = static_cast<float>(v.size());
        return std::accumulate(v.begin(), v.end(), 0.0f) / count * 1000;
    }

    static float median(std::vector<float> v) {
        size_t size = v.size();

        if (size == 0)
            return 0;

        sort(v.begin(), v.end());
        return v[size / 2] * 1000;
    }

    static float min(std::vector<float> v) {
        size_t size = v.size();

        if (size == 0)
            return 0;

        sort(v.begin(), v.end());
        return v.front() * 1000;
    }

    static float max(std::vector<float> v) {
        size_t size = v.size();

        if (size == 0)
            return 0;

        sort(v.begin(), v.end());
        return v.back() * 1000;
    }

    static double std_dev(std::vector<float> const &v) {
        if (v.empty())
            return 0;

        auto const count = static_cast<float>(v.size());
        float mean = std::accumulate(v.begin(), v.end(), 0.0f) / count;

        std::vector<double> diff(v.size());

        std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
        double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        return std::sqrt(sq_sum / count);
    }
};

template<typename S>
std::ostream &operator<<(std::ostream &os, const std::vector<S> &vector) {
    // Printing all the elements
    // using <<
    for (auto element: vector) {
        os << std::hex << std::setfill('0') << std::setw(2) << (int) element << " ";
    }
    return os;
}

class LazyCUDATimer {
public:
    void start() {
        if (!initialized_) {
            initialize();
        }
        timer_->start();
    }

    void stop() {
        if (timer_) {
            timer_->stop();
        }
    }

    LazyCUDATimer(std::string func_name){
        func_name_ = std::move(func_name);
    }

private:
    void initialize() {
        timer_ = std::make_unique<CUDATimer>(func_name_);
        initialized_ = true;
    }

    bool initialized_ = false;
    std::unique_ptr<CUDATimer> timer_;
    std::string func_name_;
};

#endif