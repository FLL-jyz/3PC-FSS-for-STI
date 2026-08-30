#pragma once

#include <algorithm>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <thread>
#include "globals.h"
#include "params.h"
#include "rss_protocols.h"
#include "ass_protocols.h"
#include "party3pc.h"
#include <thread>

namespace rss_protocols
{
    namespace parallel
    {
        inline constexpr uint64_t kParallelWorkThreshold = 1ULL << 12;

        inline int &omp_thread_limit_override()
        {
            static thread_local int thread_limit = 0;
            return thread_limit;
        }

        inline int omp_thread_limit()
        {
            const int thread_limit = omp_thread_limit_override();
            return thread_limit > 0 ? thread_limit : omp_get_max_threads();
        }

        inline int concurrent_task_thread_limit(int task_count)
        {
            if (task_count <= 1)
            {
                return omp_thread_limit();
            }
            return std::max(1, omp_thread_limit() / task_count);
        }

        class ScopedOmpThreadLimit
        {
        public:
            explicit ScopedOmpThreadLimit(int thread_limit)
                : previous_override_(omp_thread_limit_override()), previous_runtime_(omp_get_max_threads())
            {
                const int effective_limit = std::max(1, thread_limit);
                omp_thread_limit_override() = effective_limit;
                omp_set_num_threads(effective_limit);
            }

            ~ScopedOmpThreadLimit()
            {
                omp_thread_limit_override() = previous_override_;
                omp_set_num_threads(previous_runtime_);
            }

        private:
            int previous_override_;
            int previous_runtime_;
        };

        inline bool should_parallelize(uint64_t outer_work_items, uint64_t inner_work_items = 1)
        {
            return outer_work_items * inner_work_items >= kParallelWorkThreshold;
        }
    }

    namespace debug
    {
        template <typename T>
        void openPrintReal(RSSTensor<T> &x);

        template <typename T>
        void openPrintReal(RSSTensor<T> &x, int index);

        template <typename T>
        void openPrintReal(RSSTensor<T> &x, std::vector<int> index);

        template <typename T>
        void openPrintReal(RSSTensor<T> &x, int start, int end);

        template <typename T>
        void printRealToFile(RSSTensor<T> &x, const std::string &file_name);
    }

    namespace utils
    {
        constexpr uint64_t kMinOmpWorkItems = 1u << 14;

        inline bool should_parallelize(uint64_t work_items)
        {
            return work_items >= kMinOmpWorkItems;
        }

        template <typename T>
        void RSSMatMul(RSSTensor<T> &x, RSSTensor<T> &y, Tensor<T> &res, const uint32_t broadcast_size,
                       const uint32_t common_size, const uint32_t row, const uint32_t common_dim, const uint32_t col,
                       const bool is_b_broadcast);

        template <typename T>
        void RSSMatMul(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &res, const uint32_t broadcast_size,
                       const uint32_t common_size, const uint32_t row, const uint32_t common_dim, const uint32_t col,
                       const bool is_b_broadcast);

        template <typename T>
        void RSSMatMul(Tensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, const uint32_t broadcast_size,
                       const uint32_t common_size, const uint32_t row, const uint32_t common_dim, const uint32_t col,
                       const bool is_b_broadcast);

        template <typename T>
        // Strict private compare protocol (Algorithm 1):
        // compare secret shared x bits with public threshold bits r from preprocessing.
        // Input is secret x bits in RSS encoding; r is read from parameter.pc_cmp.r_*_bits.
        void privateCompare(RSSTensor<T> &secret_x_bits, RSSTensor<T> &res, Parameters<T> &parameter);

        // theta = wrap3(a1, a2, a3, L): return RSS shares in theta.
        template <typename T>
        void wrap3(RSSTensor<T> &x, RSSTensor<T> &theta, Parameters<T> &parameter, bool malicious = false);

        template <typename T>
        void getk(RSSTensor<T> &x, RSSTensor<T> &k, Parameters<T> &parameter, bool malicious = false);

        template <typename T>
        void gelu_same_scale(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious = false);
    }

    template <typename T>
    void restore(RSSTensor<T> &x, Tensor<T> &res, bool malicious = false);

    template <typename T>
    void reconstruct_to(int target_id, RSSTensor<T> &x, Tensor<T> &res = NULL, bool malicious = false);

    template <typename T>
    void share(Tensor<T> &x, RSSTensor<T> &res);

    template <typename T>
    void recv_shares_from(int source_id, RSSTensor<T> &res);

    template <typename T>
    void coin(std::vector<uint32_t> shape, RSSTensor<T> &res);

    template <typename T>
    void reshare(Tensor<T> &x, RSSTensor<T> &res);

    template <typename T>
    RSSTensor<T> reshare(Tensor<T> &x);

    template <typename T>
    void add(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res);

    template <typename T>
    void add(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &res);

    template <typename T>
    void add(RSSTensor<T> &x, T y, RSSTensor<T> &res);

    template <typename T>
    void sub(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res);

    template <typename T>
    void sub(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &res);

    template <typename T>
    void sub(RSSTensor<T> &x, T y, RSSTensor<T> &res);

    template <typename T>
    void sub(T x, RSSTensor<T> &y, RSSTensor<T> &res);

    template <typename T>
    void mulConst(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &res);

    template <typename T>
    void mulConst(RSSTensor<T> &x, T y, RSSTensor<T> &res);

    template <typename T>
    void mulConstAddBias(RSSTensor<T> &x, T y, T bias, RSSTensor<T> &res);

    template <typename T>
    void mulConstAddBias(RSSTensor<T> &x, T y, RSSTensor<T> &bias, RSSTensor<T> &res);

    template <typename T>
    void mulConstAddBias(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &bias, RSSTensor<T> &res);

    template <typename T>
    void mulConstSubBias(RSSTensor<T> &x, T y, T bias, RSSTensor<T> &res);

    template <typename T>
    void mulConstSubBias(RSSTensor<T> &x, T y, RSSTensor<T> &bias, RSSTensor<T> &res);

    template <typename T>
    void mulConstSubBias(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &bias, RSSTensor<T> &res);

    template <typename T>
    void mul(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, bool needTrunc = false, bool malicious = false);

    template <typename T>
    void mul(RSSTensor<T> &x, std::pair<T, T> &y, RSSTensor<T> &res);

    template <typename T>
    void square(RSSTensor<T> &x, RSSTensor<T> &res, bool needTrunc = false, bool malicious = false);

    template <typename T>
    void matMul(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, bool needTrunc = false, bool malicious = false);

    template <typename T>
    void truncate(RSSTensor<T> &x, RSSTensor<T> &res, size_t scale, bool malicious = false);

    template <typename T>
    void truncate(RSSTensor<T> &x, size_t scale, bool malicious = false);

    template <typename T>
    void truncate(RSSTensor<T> &x, bool malicious = false);

    template <typename T>
    void checkZero(RSSTensor<T> &x);

    template <typename T>
    void macCheck(const RSSTensor<T> &x, const RSSTensor<T> &mx, const std::pair<T, T> &mac_key);

    template <typename T>
    void macCheck(RSSTensor<T> &x, RSSTensor<T> &mx, RSSTensor<T> &mac_key);

    template <typename T>
    void macCheckSimulate(uint32_t size);

    template <typename T>
    void pc_msb(RSSTensor<T> &x, RSSTensor<T> &res,
                Parameters<T> &parameter, const uint32_t size, bool malicious = false);

    template <typename T>
    void nonNegative(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool isFloat = true, bool malicious = false);

    template <typename T>
    void greaterEqual(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, Parameters<T> &parameter, bool isFloat = true, bool malicious = false);

    template <typename T>
    void select(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, Parameters<T> &parameter, bool malicious = false);

    template <typename T>
    void select(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, uint32_t y_num, Parameters<T> &parameter, bool malicious = false);

    template <typename T>
    void inv(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious = false);

    template <typename T>
    void rsqrt(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious = false);

    template <typename T>
    void gelu(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious = false);

    template <typename T>
    void silu(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious = false);

    template <typename T>
    void rms_norm(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious = false,
                  long double eps = 1e-5L);

    template <typename T>
    void max_last_dim(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious = false);

    template <typename T>
    void neg_exp(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &Parameters, bool malicious = false);

    template <typename T>
    void softmax_forward(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool malicious = false);

    template <typename T>
    void grouped_multi_query_attention(RSSTensor<T> &query, RSSTensor<T> &key, RSSTensor<T> &value, RSSTensor<T> &res,
                                       Parameters<T> &parameter, uint32_t num_heads,
                                       uint32_t num_key_value_heads, bool masked = true, bool malicious = false);

}

template <typename T>
void rss_protocols::debug::openPrintReal(RSSTensor<T> &x)
{
    openPrintReal(x, 0, x.size());
}

template <typename T>
void rss_protocols::debug::openPrintReal(RSSTensor<T> &x, int index)
{
    openPrintReal(x, index, index + 1);
}

template <typename T>
void rss_protocols::debug::openPrintReal(RSSTensor<T> &x, std::vector<int> index)
{
    Tensor<T> real_res(x.shape);
    restore(x, real_res);

    if (Party3PC::getInstance().party_id == 0)
    {
        for (int i : index)
        {
            always_assert(i < x.size());
            std::cout << (float)(long)real_res.data[i] / (1 << x.float_scale_bit) << ", ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
void rss_protocols::debug::openPrintReal(RSSTensor<T> &x, int start, int end)
{
    always_assert(start < end);
    Tensor<T> real_res(x.shape);
    restore(x, real_res);

    if (Party3PC::getInstance().party_id == 0)
    {
        for (int i = start; i < end; i++)
        {
            if constexpr (std::is_same_v<T, uint32_t>)
            {
                std::cout << (float)(int)real_res.data[i] / (1 << x.float_scale_bit) << ", ";
            }
            else
            {
                std::cout << (double)(long)real_res.data[i] / (1 << x.float_scale_bit) << ", ";
            }
        }
        std::cout << std::endl;
    }
}

template <typename T>
void rss_protocols::debug::printRealToFile(RSSTensor<T> &x, const std::string &file_name)
{
    Tensor<T> real_res(x.shape);
    restore(x, real_res);

    if (Party3PC::getInstance().party_id == 0)
    {
        std::ofstream outFile;
        outFile.open(file_name);
        outFile << "[";
        for (int i = 0; i < x.size(); i++)
        {
            if constexpr (std::is_same_v<T, uint32_t>)
            {
                outFile << ((float)(int)real_res.data[i] / (1 << x.float_scale_bit)) << ", ";
            }
            else
            {
                outFile << ((double)(long)real_res.data[i] / (1 << x.float_scale_bit)) << ", ";
            }
        }
        outFile << "]" << std::endl;
    }
}

template <typename T>
void rss_protocols::utils::RSSMatMul(RSSTensor<T> &x, RSSTensor<T> &y, Tensor<T> &res, const uint32_t broadcast_size,
                                     const uint32_t common_size, const uint32_t row, const uint32_t common_dim, const uint32_t col,
                                     const bool is_b_broadcast)
{
    uint32_t index, index_a, index_b;
    if (is_b_broadcast)
    {
#pragma omp parallel for collapse(3) if (utils::should_parallelize(static_cast<uint64_t>(broadcast_size) * common_size * row * col))
        for (uint32_t i = 0; i < broadcast_size; i++)
        {
            for (uint32_t j = 0; j < common_size; j++)
            {
                for (uint32_t k = 0; k < row; k++)
                {
                    for (uint32_t l = 0; l < col; l++)
                    {
                        index = i * common_size * row * col + j * row * col + k * col + l;
                        index_a = i * common_size * row * common_dim + j * row * common_dim + k * common_dim;
                        index_b = j * common_dim * col + l;

                        res.data[index] = 0;
                        for (uint32_t m = 0; m < common_dim; m++)
                        {
                            res.data[index] += x.first.data[index_a + m] * y.first.data[index_b + m * col] + x.second.data[index_a + m] * y.first.data[index_b + m * col] + x.first.data[index_a + m] * y.second.data[index_b + m * col];
                        }
                    }
                }
            }
        }
    }
    else
    {
#pragma omp parallel for collapse(3) if (utils::should_parallelize(static_cast<uint64_t>(broadcast_size) * common_size * row * col))
        for (uint32_t i = 0; i < broadcast_size; i++)
        {
            for (uint32_t j = 0; j < common_size; j++)
            {
                for (uint32_t k = 0; k < row; k++)
                {
                    for (uint32_t l = 0; l < col; l++)
                    {
                        index = i * common_size * row * col + j * row * col + k * col + l;
                        index_a = j * row * common_dim + k * common_dim;
                        index_b = i * common_size * common_dim * col + j * common_dim * col + l;

                        res.data[index] = 0;
                        for (uint32_t m = 0; m < common_dim; m++)
                        {
                            res.data[index] += x.first.data[index_a + m] * y.first.data[index_b + m * col] + x.second.data[index_a + m] * y.first.data[index_b + m * col] + x.first.data[index_a + m] * y.second.data[index_b + m * col];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void rss_protocols::utils::RSSMatMul(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &res, const uint32_t broadcast_size,
                                     const uint32_t common_size, const uint32_t row, const uint32_t common_dim, const uint32_t col,
                                     const bool is_b_broadcast)
{
    uint32_t index, index_a, index_b;
    if (is_b_broadcast)
    {
#pragma omp parallel for collapse(3) if (utils::should_parallelize(static_cast<uint64_t>(broadcast_size) * common_size * row * col))
        for (uint32_t i = 0; i < broadcast_size; i++)
        {
            for (uint32_t j = 0; j < common_size; j++)
            {
                for (uint32_t k = 0; k < row; k++)
                {
                    for (uint32_t l = 0; l < col; l++)
                    {
                        index = i * common_size * row * col + j * row * col + k * col + l;
                        index_a = i * common_size * row * common_dim + j * row * common_dim + k * common_dim;
                        index_b = j * common_dim * col + l;

                        res.first.data[index] = 0;
                        res.second.data[index] = 0;
                        for (uint32_t m = 0; m < common_dim; m++)
                        {
                            res.first.data[index] += x.first.data[index_a + m] * y.data[index_b + m * col];
                            res.second.data[index] += x.second.data[index_a + m] * y.data[index_b + m * col];
                        }
                    }
                }
            }
        }
    }
    else
    {
#pragma omp parallel for collapse(3) if (utils::should_parallelize(static_cast<uint64_t>(broadcast_size) * common_size * row * col))
        for (uint32_t i = 0; i < broadcast_size; i++)
        {
            for (uint32_t j = 0; j < common_size; j++)
            {
                for (uint32_t k = 0; k < row; k++)
                {
                    for (uint32_t l = 0; l < col; l++)
                    {
                        index = i * common_size * row * col + j * row * col + k * col + l;
                        index_a = j * row * common_dim + k * common_dim;
                        index_b = i * common_size * common_dim * col + j * common_dim * col + l;

                        res.first.data[index] = 0;
                        res.second.data[index] = 0;
                        for (uint32_t m = 0; m < common_dim; m++)
                        {
                            res.first.data[index] += x.first.data[index_a + m] * y.data[index_b + m * col];
                            res.second.data[index] += x.second.data[index_a + m] * y.data[index_b + m * col];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void rss_protocols::utils::RSSMatMul(Tensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, const uint32_t broadcast_size,
                                     const uint32_t common_size, const uint32_t row, const uint32_t common_dim, const uint32_t col,
                                     const bool is_b_broadcast)
{
    uint32_t index, index_a, index_b;
    if (is_b_broadcast)
    {
#pragma omp parallel for collapse(3) if (utils::should_parallelize(static_cast<uint64_t>(broadcast_size) * common_size * row * col))
        for (uint32_t i = 0; i < broadcast_size; i++)
        {
            for (uint32_t j = 0; j < common_size; j++)
            {
                for (uint32_t k = 0; k < row; k++)
                {
                    for (uint32_t l = 0; l < col; l++)
                    {
                        index = i * common_size * row * col + j * row * col + k * col + l;
                        index_a = i * common_size * row * common_dim + j * row * common_dim + k * common_dim;
                        index_b = j * common_dim * col + l;

                        res.first.data[index] = 0;
                        res.second.data[index] = 0;
                        for (uint32_t m = 0; m < common_dim; m++)
                        {
                            res.first.data[index] += x.data[index_a + m] * y.first.data[index_b + m * col];
                            res.second.data[index] += x.data[index_a + m] * y.second.data[index_b + m * col];
                        }
                    }
                }
            }
        }
    }
    else
    {
#pragma omp parallel for collapse(3) if (utils::should_parallelize(static_cast<uint64_t>(broadcast_size) * common_size * row * col))
        for (uint32_t i = 0; i < broadcast_size; i++)
        {
            for (uint32_t j = 0; j < common_size; j++)
            {
                for (uint32_t k = 0; k < row; k++)
                {
                    for (uint32_t l = 0; l < col; l++)
                    {
                        index = i * common_size * row * col + j * row * col + k * col + l;
                        index_a = j * row * common_dim + k * common_dim;
                        index_b = i * common_size * common_dim * col + j * common_dim * col + l;

                        res.first.data[index] = 0;
                        res.second.data[index] = 0;
                        for (uint32_t m = 0; m < common_dim; m++)
                        {
                            res.first.data[index] += x.data[index_a + m] * y.first.data[index_b + m * col];
                            res.second.data[index] += x.data[index_a + m] * y.second.data[index_b + m * col];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void rss_protocols::restore(RSSTensor<T> &x, Tensor<T> &res, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    x.assert_same_shape(res);
    
    std::thread t([&]() {
        party.send_tensor_to(party.next_party_id, x.first);
    });

    party.recv_tensor_from(party.pre_party_id, res);
    t.join();

    if (malicious)
    {
        std::thread t2([&]() {
            party.send_tensor_to(party.pre_party_id, x.second);
        });
        
        Tensor<T> tmp(x.shape);
        party.recv_tensor_from(party.next_party_id, tmp);
        t2.join();

        Tensor<T>::minus(tmp, res, tmp);
        tmp.assert_zero();
    }

#pragma omp parallel for if (utils::should_parallelize(x.size()))
    for (int i = 0; i < x.size(); i++)
    {
        res.data[i] = res.data[i] + x.first.data[i] + x.second.data[i];
    }
}

template <typename T>
void rss_protocols::reconstruct_to(int target_id, RSSTensor<T> &x, Tensor<T> &res, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    if (target_id == party.pre_party_id)
    {
        if (malicious)
        {   
            std::thread t1([&](){ party.send_tensor_to(target_id, x.second); });
            t1.join();
        }
    }
    else if (target_id == party.next_party_id)
    {   
        std::thread t1([&](){ party.send_tensor_to(target_id, x.first); });
        t1.join();
    }
    else
    {
        Tensor<T> tmp(x.shape);
        std::thread t1([&](){ party.recv_tensor_from(party.pre_party_id, res); });
        t1.join();
        if (malicious)
        {   
            std::thread t2([&](){ party.recv_tensor_from(party.next_party_id, tmp); });
            t2.join();

            Tensor<T>::minus(tmp, tmp, res);
            tmp.assert_zero();
        }

#pragma omp parallel for if (utils::should_parallelize(x.size()))
        for (int i = 0; i < x.size(); i++)
        {
            res.data[i] = res.data[i] + x.first.data[i] + x.second.data[i];
        }
    }
}

template <typename T>
void rss_protocols::share(Tensor<T> &x, RSSTensor<T> &res)
{
    Party3PC &party = Party3PC::getInstance();
    res.assert_same_shape(x);
    res.rand();
    Tensor<T> tmp(x.shape);
#pragma omp parallel for if (utils::should_parallelize(x.size()))
    for (int i = 0; i < x.size(); i++)
    {
        tmp.data[i] = x.data[i] - res.first.data[i] - res.second.data[i];
    }

    std::thread t1([&](){
        party.send_tensor_to(party.next_party_id, res.second);
        party.send_tensor_to(party.next_party_id, tmp);
    });
    
    std::thread t2([&](){
        party.send_tensor_to(party.pre_party_id, tmp); 
        party.send_tensor_to(party.pre_party_id, res.first); 
    });

    t1.join();
    t2.join();
    

}

template <typename T>
void rss_protocols::recv_shares_from(int source_id, RSSTensor<T> &res)
{
    Party3PC &party = Party3PC::getInstance();
    party.recv_tensor_from(source_id, res.first);
    party.recv_tensor_from(source_id, res.second);
}

template <typename T>
void rss_protocols::coin(std::vector<uint32_t> shape, RSSTensor<T> &res)
{
}

template <typename T>
void rss_protocols::reshare(Tensor<T> &x, RSSTensor<T> &res)
{
    Party3PC &party = Party3PC::getInstance();
    res.assert_same_shape(x);
    std::thread t1([&](){ party.send_tensor_to(party.pre_party_id, x); });
    party.recv_tensor_from(party.next_party_id, res.second);
    t1.join();
    res.first.copy(x);
}

template <typename T>
RSSTensor<T> rss_protocols::reshare(Tensor<T> &x)
{
    Party3PC &party = Party3PC::getInstance();
    Tensor<T> tmp(x.shape);
    std::thread t1([&](){ party.send_tensor_to(party.pre_party_id, x); });
    party.recv_tensor_from(party.next_party_id, tmp);
    t1.join();
    return RSSTensor<T>(x, tmp);
}

template <typename T>
void rss_protocols::add(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(y);

#pragma omp parallel for if (utils::should_parallelize(x.size()))
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = x.first.data[i] + y.first.data[i];
        res.second.data[i] = x.second.data[i] + y.second.data[i];
    }
}

template <typename T>
void rss_protocols::add(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &res)
{
    int party_id = Party3PC::getInstance().party_id;
    res.assert_same_shape(x);
    res.assert_same_shape(y);

    if (party_id == 0)
    {
#pragma omp parallel for if (utils::should_parallelize(x.size()))
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] + y.data[i];
            res.second.data[i] = x.second.data[i];
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for if (utils::should_parallelize(x.size()))
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i];
        }
    }
    else
    {
#pragma omp parallel for if (utils::should_parallelize(x.size()))
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i] + y.data[i];
        }
    }
}

template <typename T>
void rss_protocols::add(RSSTensor<T> &x, T y, RSSTensor<T> &res)
{
    int party_id = Party3PC::getInstance().party_id;
    res.assert_same_shape(x);

    if (party_id == 0)
    {
#pragma omp parallel for if (utils::should_parallelize(x.size()))
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] + y;
            res.second.data[i] = x.second.data[i];
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for if (utils::should_parallelize(x.size()))
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i];
        }
    }
    else
    {
#pragma omp parallel for if (utils::should_parallelize(x.size()))
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i] + y;
        }
    }
}

template <typename T>
void rss_protocols::sub(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(y);

#pragma omp parallel for if (utils::should_parallelize(x.size()))
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = x.first.data[i] - y.first.data[i];
        res.second.data[i] = x.second.data[i] - y.second.data[i];
    }
}

template <typename T>
void rss_protocols::sub(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &res)
{
    int party_id = Party3PC::getInstance().party_id;
    res.assert_same_shape(x);
    res.assert_same_shape(y);

    if (party_id == 0)
    {
#pragma omp parallel for if (utils::should_parallelize(x.size()))
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] - y.data[i];
            res.second.data[i] = x.second.data[i];
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for if (utils::should_parallelize(x.size()))
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i];
        }
    }
    else
    {
#pragma omp parallel for if (utils::should_parallelize(x.size()))
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i] - y.data[i];
        }
    }
}

template <typename T>
void rss_protocols::sub(RSSTensor<T> &x, T y, RSSTensor<T> &res)
{
    int party_id = Party3PC::getInstance().party_id;
    res.assert_same_shape(x);

    if (party_id == 0)
    {
#pragma omp parallel for if (utils::should_parallelize(x.size()))
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] - y;
            res.second.data[i] = x.second.data[i];
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for if (utils::should_parallelize(x.size()))
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i];
        }
    }
    else
    {
#pragma omp parallel for if (utils::should_parallelize(x.size()))
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i] - y;
        }
    }
}

template <typename T>
void rss_protocols::sub(T x, RSSTensor<T> &y, RSSTensor<T> &res)
{
    int party_id = Party3PC::getInstance().party_id;
    res.assert_same_shape(y);

    if (party_id == 0)
    {
#pragma omp parallel for if (utils::should_parallelize(y.size()))
        for (int i = 0; i < y.size(); i++)
        {
            res.first.data[i] = x - y.first.data[i];
            res.second.data[i] = -y.second.data[i];
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for if (utils::should_parallelize(y.size()))
        for (int i = 0; i < y.size(); i++)
        {
            res.first.data[i] = -y.first.data[i];
            res.second.data[i] = -y.second.data[i];
        }
    }
    else
    {
#pragma omp parallel for if (utils::should_parallelize(y.size()))
        for (int i = 0; i < y.size(); i++)
        {
            res.first.data[i] = -y.first.data[i];
            res.second.data[i] = x - y.second.data[i];
        }
    }
}

template <typename T>
void rss_protocols::mulConst(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(y);

#pragma omp parallel for if (utils::should_parallelize(x.size()))
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = x.first.data[i] * y.data[i];
        res.second.data[i] = x.second.data[i] * y.data[i];
    }
}

template <typename T>
void rss_protocols::mulConst(RSSTensor<T> &x, T y, RSSTensor<T> &res)
{
    res.assert_same_shape(x);

#pragma omp parallel for if (utils::should_parallelize(x.size()))
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = x.first.data[i] * y;
        res.second.data[i] = x.second.data[i] * y;
    }
}

template <typename T>
void rss_protocols::mulConstAddBias(RSSTensor<T> &x, T y, T bias, RSSTensor<T> &res)
{
    int party_id = Party3PC::getInstance().party_id;
    res.assert_same_shape(x);

    if (party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] * y + bias;
            res.second.data[i] = x.second.data[i] * y;
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] * y;
            res.second.data[i] = x.second.data[i] * y;
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] * y;
            res.second.data[i] = x.second.data[i] * y + bias;
        }
    }
}

template <typename T>
void rss_protocols::mulConstAddBias(RSSTensor<T> &x, T y, RSSTensor<T> &bias, RSSTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(bias);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = x.first.data[i] * y + bias.first.data[i];
        res.second.data[i] = x.second.data[i] * y + bias.second.data[i];
    }
}

template <typename T>
void rss_protocols::mulConstAddBias(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &bias, RSSTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(y);
    res.assert_same_shape(bias);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = x.first.data[i] * y.data[i] + bias.first.data[i];
        res.second.data[i] = x.second.data[i] * y.data[i] + bias.second.data[i];
    }
}

template <typename T>
void rss_protocols::mulConstSubBias(RSSTensor<T> &x, T y, T bias, RSSTensor<T> &res)
{
    int party_id = Party3PC::getInstance().party_id;
    res.assert_same_shape(x);

    if (party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] * y - bias;
            res.second.data[i] = x.second.data[i] * y;
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] * y;
            res.second.data[i] = x.second.data[i] * y;
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] * y;
            res.second.data[i] = x.second.data[i] * y - bias;
        }
    }
}

template <typename T>
void rss_protocols::mulConstSubBias(RSSTensor<T> &x, T y, RSSTensor<T> &bias, RSSTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(bias);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = x.first.data[i] * y - bias.first.data[i];
        res.second.data[i] = x.second.data[i] * y - bias.second.data[i];
    }
}

template <typename T>
void rss_protocols::mulConstSubBias(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &bias, RSSTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(bias);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = x.first.data[i] * y.data[i] - bias.first.data[i];
        res.second.data[i] = x.second.data[i] * y.data[i] - bias.second.data[i];
    }
}

template <typename T>
void rss_protocols::mul(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, bool needTrunc, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    res.assert_same_shape(x);
    res.assert_same_shape(y);
    uint32_t size = x.size();

    Tensor<T> tmp(x.shape);
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        tmp.data[i] = x.first.data[i] * y.first.data[i] + x.first.data[i] * y.second.data[i] +
                      x.second.data[i] * y.first.data[i];
    }

    reshare(tmp, res);
    if (malicious)
    {
        RSSTensor<T> a(x.shape), b(y.shape), c(res.shape);
        a.zeros();
        b.zeros();
        c.zeros();

        uint32_t combined_size = 2 * size;
        RSSTensor<T> combined({combined_size});

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            combined.first.data[i] = a.first.data[i] - x.first.data[i];
            combined.second.data[i] = a.second.data[i] - x.second.data[i];
            combined.first.data[i + size] = b.first.data[i] - y.first.data[i];
            combined.second.data[i + size] = b.second.data[i] - y.second.data[i];
        }
        Tensor<T> rhoSigma({combined_size}), rho(x.shape), sigma(y.shape);
        restore(combined, rhoSigma, malicious);

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            rho.data[i] = rhoSigma.data[i];
            sigma.data[i] = rhoSigma.data[i + size];
        }

        RSSTensor<T> zero_res(x.shape);

        if (party.party_id == 0)
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + a.first.data[i] * sigma.data[i] +
                                         rho.data[i] * b.first.data[i] - rho.data[i] * sigma.data[i];
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + a.second.data[i] * sigma.data[i] +
                                          rho.data[i] * b.second.data[i];
            }
        }
        else if (party.party_id == 1)
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + a.first.data[i] * sigma.data[i] +
                                         rho.data[i] * b.first.data[i];
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + a.second.data[i] * sigma.data[i] +
                                          rho.data[i] * b.second.data[i];
            }
        }
        else
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + a.first.data[i] * sigma.data[i] +
                                         rho.data[i] * b.first.data[i];
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + a.second.data[i] * sigma.data[i] +
                                          rho.data[i] * b.second.data[i] - rho.data[i] * sigma.data[i];
            }
        }

        std::thread t1([&](){ party.send_tensor_to(party.next_party_id, zero_res.first); });
        t1.join();
        Tensor<T> tmp_recv(x.shape);
        std::thread t2([&](){ party.recv_tensor_from(party.pre_party_id, tmp_recv); });
        t2.join();
        // party.send_tensor_to(party.next_party_id, zero_res.first);
        // Tensor<T> tmp_recv(x.shape);
        // party.recv_tensor_from(party.pre_party_id, tmp_recv);

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            always_assert(zero_res.first.data[i] + zero_res.second.data[i] + tmp_recv.data[i] == 0);
        }
    }
    if (needTrunc)
    {
        truncate(res, malicious);
    }
}

template <typename T>
void rss_protocols::mul(RSSTensor<T> &x, std::pair<T, T> &y, RSSTensor<T> &res)
{
    res.assert_same_shape(x);
    uint32_t size = x.size();

    Tensor<T> tmp(x.shape);
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        tmp.data[i] = x.first.data[i] * y.first + x.first.data[i] * y.second +
                      x.second.data[i] * y.first;
    }

    reshare(tmp, res);
}

template <typename T>
void rss_protocols::square(RSSTensor<T> &x, RSSTensor<T> &res, bool needTrunc, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    res.assert_same_shape(x);
    uint32_t size = x.size();

    Tensor<T> tmp(x.shape);
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        tmp.data[i] = x.first.data[i] * x.first.data[i] + x.first.data[i] * x.second.data[i] +
                      x.second.data[i] * x.first.data[i];
    }

    reshare(tmp, res);
    if (malicious)
    {
        RSSTensor<T> a(x.shape), c(res.shape); // c = a * a
        a.zeros();
        c.zeros();

        RSSTensor<T> rho_share({size});

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            rho_share.first.data[i] = a.first.data[i] - x.first.data[i];
            rho_share.second.data[i] = a.second.data[i] - x.second.data[i];
        }
        Tensor<T> rho({size});
        restore(rho_share, rho, malicious);

        RSSTensor<T> zero_res(x.shape);

        if (party.party_id == 0)
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + a.first.data[i] * rho.data[i] * 2 - rho.data[i] * rho.data[i];
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + a.second.data[i] * rho.data[i] * 2;
            }
        }
        else if (party.party_id == 1)
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + a.first.data[i] * rho.data[i] * 2;
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + a.second.data[i] * rho.data[i] * 2;
            }
        }
        else
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + a.first.data[i] * rho.data[i] * 2;
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + a.second.data[i] * rho.data[i] * 2 - rho.data[i] * rho.data[i];
            }
        }

        std::thread t1([&](){ party.send_tensor_to(party.next_party_id, zero_res.first); });
        t1.join();
        Tensor<T> tmp_recv(x.shape);
        std::thread t2([&](){ party.recv_tensor_from(party.pre_party_id, tmp_recv); });
        t2.join();

        // party.send_tensor_to(party.next_party_id, zero_res.first);
        // Tensor<T> tmp_recv(x.shape);
        // party.recv_tensor_from(party.pre_party_id, tmp_recv);

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            always_assert(zero_res.first.data[i] + zero_res.second.data[i] + tmp_recv.data[i] == 0);
        }
    }
    if (needTrunc)
    {
        truncate(res, malicious);
    }
}

template <typename T>
void rss_protocols::matMul(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, bool needTrunc, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    uint32_t size = res.size(), x_size = x.size(), y_size = y.size();
    Tensor<T> tmp(res.shape);

    const uint32_t x_shape_size = x.shape.size();
    const uint32_t y_shape_size = y.shape.size();
    const uint32_t z_shape_size = res.shape.size();
    always_assert(x_shape_size >= 2 && y_shape_size >= 2 && z_shape_size >= 2);
    always_assert(x.shape[x_shape_size - 1] == y.shape[y_shape_size - 2]);
    always_assert(res.shape[z_shape_size - 2] == x.shape[x_shape_size - 2]);
    always_assert(res.shape[z_shape_size - 1] == y.shape[y_shape_size - 1]);

    uint32_t row, common_dim, col, broadcast_size, common_size;
    bool is_b_broadcast;

    compute_matmul_info(x.shape, y.shape, res.shape, row, common_dim, col, broadcast_size, common_size,
                        is_b_broadcast);

    utils::RSSMatMul(x, y, tmp, broadcast_size, common_size, row, common_dim, col, is_b_broadcast);
    reshare(tmp, res);
    if (malicious)
    {
        RSSTensor<T> a(x.shape), b(y.shape), c(res.shape);
        a.zeros();
        b.zeros();
        c.zeros();

        uint32_t combined_size = x_size + y_size;
        RSSTensor<T> combined({combined_size});

#pragma omp parallel for if (utils::should_parallelize(x_size))
        for (int i = 0; i < x_size; i++)
        {
            combined.first.data[i] = a.first.data[i] - x.first.data[i];
            combined.second.data[i] = a.second.data[i] - x.second.data[i];
        }
#pragma omp parallel for if (utils::should_parallelize(y_size))
        for (int i = 0; i < y_size; i++)
        {
            combined.first.data[i + x_size] = b.first.data[i] - y.first.data[i];
            combined.second.data[i + x_size] = b.second.data[i] - y.second.data[i];
        }

        Tensor<T> rhoSigma({combined_size}), rho(x.shape), sigma(y.shape);
        restore(combined, rhoSigma, malicious);

#pragma omp parallel for if (utils::should_parallelize(combined_size))
        for (int i = 0; i < x_size; i++)
        {
            rho.data[i] = rhoSigma.data[i];
        }
        for (int i = 0; i < y_size; i++)
        {
            sigma.data[i] = rhoSigma.data[i + x_size];
        }

        RSSTensor<T> zero_res(res.shape), af(res.shape), eb(res.shape);
        Tensor<T> ef(res.shape);

        utils::RSSMatMul(a, sigma, af, broadcast_size, common_size, row, common_dim, col, is_b_broadcast);
        utils::RSSMatMul(rho, b, eb, broadcast_size, common_size, row, common_dim, col, is_b_broadcast);
        Tensor<T>::matmul(ef, rho, sigma, broadcast_size, common_size, row, common_dim, col, is_b_broadcast);

        if (party.party_id == 0)
        {
#pragma omp parallel for if (utils::should_parallelize(size))
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] =
                    res.first.data[i] - c.first.data[i] + af.first.data[i] + eb.first.data[i] - ef.data[i];
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + af.second.data[i] + eb.second.data[i];
            }
        }
        else if (party.party_id == 1)
        {
#pragma omp parallel for if (utils::should_parallelize(size))
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + af.first.data[i] + eb.first.data[i];
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + af.second.data[i] + eb.second.data[i];
            }
        }
        else
        {
#pragma omp parallel for if (utils::should_parallelize(size))
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + af.first.data[i] + eb.first.data[i];
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + af.second.data[i] + eb.second.data[i] - ef.data[i];
            }
        }
        party.send_tensor_to(party.next_party_id, zero_res.first);
        Tensor<T> tmp_recv(res.shape);
        party.recv_tensor_from(party.pre_party_id, tmp_recv);
#pragma omp parallel for if (utils::should_parallelize(size))
        for (int i = 0; i < size; i++)
        {
            always_assert(zero_res.first.data[i] + zero_res.second.data[i] + tmp_recv.data[i] == 0);
        }
    }
    if (needTrunc)
    {   
        truncate(res, malicious);
    }
}

template <typename T>
void rss_protocols::truncate(RSSTensor<T> &x, RSSTensor<T> &res, size_t scale, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    RSSTensor<T> r(x.shape), r_t(x.shape);
    uint32_t size = x.size();
    r.zeros();
    r_t.zeros();

    sub(x, r, x);
    Tensor<T> x_shift(x.shape);
    restore(x, x_shift, malicious);
    if (party.party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            if constexpr (std::is_same_v<T, uint32_t>)
            {
                res.first.data[i] =
                    r_t.first.data[i] + (T)((double)((int32_t)x_shift.data[i]) / (double)((int32_t)scale));
            }
            else
            {
                res.first.data[i] =
                    r_t.first.data[i] + (T)((double)((int64_t)x_shift.data[i]) / (double)((int64_t)scale));
            }

            res.second.data[i] = r_t.second.data[i];
        }
    }
    else if (party.party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            res.first.data[i] = r_t.first.data[i];
            res.second.data[i] = r_t.second.data[i];
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            res.first.data[i] = r_t.first.data[i];
            if constexpr (std::is_same_v<T, uint32_t>)
            {
                res.second.data[i] =
                    r_t.second.data[i] + (T)((double)((int32_t)x_shift.data[i]) / (double)((int32_t)scale));
            }
            else
            {
                res.second.data[i] =
                    r_t.second.data[i] + (T)((double)((int64_t)x_shift.data[i]) / (double)((int64_t)scale));
            }
        }
    }
}

template <typename T>
void rss_protocols::truncate(RSSTensor<T> &x, size_t scale, bool malicious)
{
    truncate(x, x, scale, malicious);
}

template <typename T>
void rss_protocols::truncate(RSSTensor<T> &x, bool malicious)
{
    truncate(x, 1 << x.float_scale_bit, malicious);
}

template <typename T>
void rss_protocols::checkZero(RSSTensor<T> &x)
{
    RSSTensor<T> r(x.shape), xr(x.shape);
    Tensor<T> xr_open(x.shape);

    r.zeros(); // it should be random
    mul(x, r, xr);
    restore(xr, xr_open, true);

    xr_open.assert_zero();
}

template <typename T>
void rss_protocols::macCheck(const RSSTensor<T> &x, const RSSTensor<T> &mx, const std::pair<T, T> &mac_key)
{
#if (IS_MALICIOUS)
    Party3PC &party = Party3PC::getInstance();
    RSSTensor<T> r({1}), mr({1});
    r.zeros(); // it should be random
    mul(r, mac_key, mr);
    RSSTensor<T> ro_share(x.shape);
    ro_share.zeros(); // it should be random
    Tensor<T> ro(x.shape);
    restore(ro_share, ro, true);
    RSSTensor<T> v({1}), w({1});
    v.first.data[0] = r.first.data[0];
    v.second.data[0] = r.second.data[0];

    w.first.data[0] = mr.first.data[0];
    w.second.data[0] = mr.second.data[0];

    for (int i = 0; i < x.size(); i++)
    {
        v.first.data[i] += x.first.data[i] * ro.data[i];
        v.second.data[i] += x.second.data[i] * ro.data[i];
        w.first.data[i] += mx.first.data[i] * ro.data[i];
        w.second.data[i] += mx.second.data[i] * ro.data[i];
    }
    Tensor<T> v_real({1});
    restore(v, v_real, true);
    RSSTensor<T> delta({1});
    mulConstSubBias(mac_key, v_real, w, delta);
    checkZero(delta);
#endif
}

template <typename T>
void rss_protocols::macCheck(RSSTensor<T> &x, RSSTensor<T> &mx, RSSTensor<T> &mac_key)
{
#if (IS_MALICIOUS)
    RSSTensor<T> r(x.shape), mr(mx.shape);
    r.zeros(); // it should be random
    mul(r, mac_key, mr);
    RSSTensor<T> ro_share(x.shape);
    ro_share.zeros(); // it should be random
    Tensor<T> ro(x.shape);
    restore(ro_share, ro, true);
    RSSTensor<T> v(x.shape), w(x.shape);
    mulConstAddBias(x, ro, r, v);
    mulConstAddBias(mx, ro, mr, w);
    Tensor<T> v_real(x.shape);
    restore(v, v_real, true);
    RSSTensor<T> delta(x.shape);
    mulConstSubBias(mac_key, v_real, w, delta);
    checkZero(delta);
#endif
}

template <typename T>
void rss_protocols::macCheckSimulate(uint32_t size)
{
#if (IS_MALICIOUS)
    if (size == 0)
    {
        return;
    }
    RSSTensor<T> x({size}), mx({size}), mac_key({size});
    x.zeros();
    mx.zeros();
    mac_key.zeros();

    macCheck(x, mx, mac_key);
    MAC_SIZE = 0;
#endif
}

template <typename T>
void rss_protocols::utils::privateCompare(RSSTensor<T> &secret_x_bits, RSSTensor<T> &res, Parameters<T> &parameter)
{
    Party3PC &party = Party3PC::getInstance();
    const uint32_t bit_length = 8 * sizeof(T);
    always_assert(secret_x_bits.size() % bit_length == 0);
    const uint32_t size = secret_x_bits.size() / bit_length;

    // Expand preprocessed public threshold bits r into the batch layout.
    // Support both single-value bit vectors (length=bit_length) and batched vectors (length=size*bit_length).
    const uint32_t r_bits_total = parameter.pc_cmp.r_with_pre_bits.size();
    always_assert(r_bits_total == bit_length || r_bits_total == secret_x_bits.size());

    RSSTensor<T> public_r_bits(secret_x_bits.shape);
#pragma omp parallel for if (utils::should_parallelize(secret_x_bits.size()))
    for (int idx = 0; idx < static_cast<int>(secret_x_bits.size()); ++idx)
    {
        const uint32_t uidx = static_cast<uint32_t>(idx);
        const uint32_t src = (r_bits_total == bit_length) ? (uidx % bit_length) : uidx;
        public_r_bits.first.data[idx] = parameter.pc_cmp.r_with_pre_bits.data[src];
        public_r_bits.second.data[idx] = parameter.pc_cmp.r_with_next_bits.data[src];
    }

    // u[i] = (-1)^beta * (r[i] - x[i]).
    RSSTensor<T> diff_bits(secret_x_bits.shape);
    sub(public_r_bits, secret_x_bits, diff_bits);

    std::pair<T, T> two_beta_minus_one = {parameter.pc_cmp.beta_with_pre * 2, parameter.pc_cmp.beta_with_next * 2};
    if (party.party_id == 0)
    {
        two_beta_minus_one.first -= 1;
    }
    else if (party.party_id == 2)
    {
        two_beta_minus_one.second -= 1;
    }

    RSSTensor<T> u_bits(secret_x_bits.shape);
    mul(diff_bits, two_beta_minus_one, u_bits);

    // w[i] = -(r[i] xor x[i]) over Z_p.
    RSSTensor<T> x_mul_r(secret_x_bits.shape), two_x_mul_r(secret_x_bits.shape), x_xor_r(secret_x_bits.shape), w_bits(secret_x_bits.shape);
    mul(public_r_bits, secret_x_bits, x_mul_r);
    mulConst(x_mul_r, static_cast<T>(2), two_x_mul_r);
    add(public_r_bits, secret_x_bits, x_xor_r);
    sub(x_xor_r, two_x_mul_r, x_xor_r);
    sub(static_cast<T>(0), x_xor_r, w_bits);

    // c[i] = u[i] + 1 + sum_{k=i+1}^{l-1} w[k] within each element bit-slice.
    RSSTensor<T> c_bits(secret_x_bits.shape);
#pragma omp parallel for if (utils::should_parallelize(secret_x_bits.size()))
    for (int s = 0; s < static_cast<int>(size); ++s)
    {
        T suffix_pre = 0;
        T suffix_next = 0;
        for (int b = static_cast<int>(bit_length) - 1; b >= 0; --b)
        {
            const uint32_t index = static_cast<uint32_t>(s) * bit_length + static_cast<uint32_t>(b);
            c_bits.first.data[index] = u_bits.first.data[index] + suffix_pre;
            c_bits.second.data[index] = u_bits.second.data[index] + suffix_next;
            if (party.party_id == 0)
            {
                c_bits.first.data[index] += 1;
            }
            else if (party.party_id == 2)
            {
                c_bits.second.data[index] += 1;
            }
            suffix_pre += w_bits.first.data[index];
            suffix_next += w_bits.second.data[index];
        }
    }

    // d = [m]^p * product_i c[i], then reveal d.
    RSSTensor<T> prod_share({size});
    for (uint32_t s = 0; s < size; ++s)
    {
        prod_share.first.data[s] = c_bits.first.data[s * bit_length];
        prod_share.second.data[s] = c_bits.second.data[s * bit_length];
    }

    for (uint32_t b = 1; b < bit_length; ++b)
    {
        RSSTensor<T> c_at_bit({size});
        for (uint32_t s = 0; s < size; ++s)
        {
            const uint32_t idx = s * bit_length + b;
            c_at_bit.first.data[s] = c_bits.first.data[idx];
            c_at_bit.second.data[s] = c_bits.second.data[idx];
        }
        mul(prod_share, c_at_bit, prod_share);
    }

    std::pair<T, T> m_share = {parameter.pc_cmp.m_with_pre, parameter.pc_cmp.m_with_next};
    RSSTensor<T> d_share({size});
    mul(prod_share, m_share, d_share);

    Tensor<T> d_open({size});
    restore(d_share, d_open);

    // Output shares of beta xor beta'. beta' = 1 iff d != 0.
    res.allocate({size});
#pragma omp parallel for if (utils::should_parallelize(size))
    for (int i = 0; i < static_cast<int>(size); ++i)
    {
        res.first.data[i] = parameter.pc_cmp.beta_with_pre;
        res.second.data[i] = parameter.pc_cmp.beta_with_next;
        const bool beta_prime = (d_open.data[i] != 0);
        if (beta_prime)
        {
            if (party.party_id == 0)
            {
                res.first.data[i] = 1 - res.first.data[i];
            }
            else if (party.party_id == 2)
            {
                res.second.data[i] = 1 - res.second.data[i];
            }
        }
    }
}


template <typename T>
void rss_protocols::utils::wrap3(RSSTensor<T> &x, RSSTensor<T> &theta, Parameters<T> &parameter, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    const uint32_t size = x.size();
    const uint32_t bit_length = 8 * sizeof(T);

    auto xor_share = [&](RSSTensor<T> &a, RSSTensor<T> &b, RSSTensor<T> &out)
    {
        RSSTensor<T> t1(x.shape), t2(x.shape), t3(x.shape);
        add(a, b, t1);
        mul(a, b, t2);
        mulConst(t2, static_cast<T>(2), t3);
        sub(t1, t3, out);
    };

    // Step 1: local masked values r_j = (a_j + x_j) mod L and local beta_j (wrap2).
    RSSTensor<T> r0_masked(x.shape), r1_masked(x.shape), r2_masked(x.shape);
    Tensor<T> beta_local(x.shape);

    const auto carry_add = [&](T a, T b) -> T
    {
        using U128 = unsigned __int128;
        const U128 sum = static_cast<U128>(a) + static_cast<U128>(b);
        return static_cast<T>((sum >> bit_length) & static_cast<U128>(1));
    };

    if (party.party_id == 0)
    {
#pragma omp parallel for if (utils::should_parallelize(size))
        for (int i = 0; i < static_cast<int>(size); i++)
        {
            r0_masked.first.data[i] = parameter.pc_cmp.self_x1 + x.first.data[i];
            r0_masked.second.data[i] = parameter.pc_cmp.self_x0 + x.second.data[i];

            r1_masked.first.data[i] = x.first.data[i];
            r1_masked.second.data[i] = parameter.pc_cmp.x_with_pre + x.second.data[i];

            r2_masked.first.data[i] = parameter.pc_cmp.x_with_next + x.first.data[i];
            r2_masked.second.data[i] = x.second.data[i];

            beta_local.data[i] = carry_add(x.first.data[i], parameter.pc_cmp.self_x1);
        }
    }
    else if (party.party_id == 1)
    {
#pragma omp parallel for if (utils::should_parallelize(size))
        for (int i = 0; i < static_cast<int>(size); i++)
        {
            r0_masked.first.data[i] = parameter.pc_cmp.x_with_next + x.first.data[i];
            r0_masked.second.data[i] = x.second.data[i];

            r1_masked.first.data[i] = parameter.pc_cmp.self_x1 + x.first.data[i];
            r1_masked.second.data[i] = parameter.pc_cmp.self_x0 + x.second.data[i];

            r2_masked.first.data[i] = x.first.data[i];
            r2_masked.second.data[i] = parameter.pc_cmp.x_with_pre + x.second.data[i];

            beta_local.data[i] = carry_add(x.first.data[i], parameter.pc_cmp.self_x1);
        }
    }
    else
    {
#pragma omp parallel for if (utils::should_parallelize(size))
        for (int i = 0; i < static_cast<int>(size); i++)
        {
            r0_masked.first.data[i] = x.first.data[i];
            r0_masked.second.data[i] = parameter.pc_cmp.x_with_pre + x.second.data[i];

            r1_masked.first.data[i] = parameter.pc_cmp.x_with_next + x.first.data[i];
            r1_masked.second.data[i] = x.second.data[i];

            r2_masked.first.data[i] = parameter.pc_cmp.self_x1 + x.first.data[i];
            r2_masked.second.data[i] = parameter.pc_cmp.self_x0 + x.second.data[i];

            beta_local.data[i] = carry_add(x.first.data[i], parameter.pc_cmp.self_x1);
        }
    }

    RSSTensor<T> beta_sum(x.shape);
    reshare(beta_local, beta_sum);

    // Step 2: explicitly recover public r_j and r.
    Tensor<T> r0_open(x.shape), r1_open(x.shape), r2_open(x.shape);
    restore(r0_masked, r0_open, malicious);
    restore(r1_masked, r1_open, malicious);
    restore(r2_masked, r2_open, malicious);

    Tensor<T> r_public(x.shape), r_plus_one_public(x.shape), delta_public({size});
#pragma omp parallel for if (utils::should_parallelize(size))
    for (int i = 0; i < static_cast<int>(size); ++i)
    {
        using U128 = unsigned __int128;
        const U128 sum_r = static_cast<U128>(r0_open.data[i]) + static_cast<U128>(r1_open.data[i]) + static_cast<U128>(r2_open.data[i]);
        r_public.data[i] = static_cast<T>(sum_r);
        r_plus_one_public.data[i] = static_cast<T>(r_public.data[i] + static_cast<T>(1));
        delta_public.data[i] = static_cast<T>((sum_r >> bit_length) & static_cast<U128>(1));
    }

    // Step 3/4: write public r+1 bits into preprocessing fields,
    // then compare secret x bits with public r+1 to obtain eta.
    parameter.pc_cmp.r_with_pre_bits.allocate({size, bit_length});
    parameter.pc_cmp.r_with_next_bits.allocate({size, bit_length});
#pragma omp parallel for if (utils::should_parallelize(size * bit_length))
    for (int i = 0; i < static_cast<int>(size); ++i)
    {
        for (uint32_t b = 0; b < bit_length; ++b)
        {
            const uint32_t idx = static_cast<uint32_t>(i) * bit_length + b;
            const T bit = static_cast<T>((r_plus_one_public.data[i] >> (bit_length - 1 - b)) & 1);
            if (party.party_id == 0)
            {
                parameter.pc_cmp.r_with_pre_bits.data[idx] = bit;
                parameter.pc_cmp.r_with_next_bits.data[idx] = 0;
            }
            else if (party.party_id == 1)
            {
                parameter.pc_cmp.r_with_pre_bits.data[idx] = 0;
                parameter.pc_cmp.r_with_next_bits.data[idx] = 0;
            }
            else
            {
                parameter.pc_cmp.r_with_pre_bits.data[idx] = 0;
                parameter.pc_cmp.r_with_next_bits.data[idx] = bit;
            }
        }
    }

    RSSTensor<T> x_bits({size, bit_length});
#pragma omp parallel for if (utils::should_parallelize(x_bits.size()))
    for (int idx = 0; idx < static_cast<int>(x_bits.size()); ++idx)
    {
        const uint32_t bit_idx = static_cast<uint32_t>(idx) % bit_length;
        x_bits.first.data[idx] = parameter.pc_cmp.x_with_pre_bits.data[bit_idx];
        x_bits.second.data[idx] = parameter.pc_cmp.x_with_next_bits.data[bit_idx];
    }

    RSSTensor<T> eta(x.shape);
    utils::privateCompare(x_bits, eta, parameter);
    eta.reshape(x.shape);

    // Step 5: theta = beta1 + beta2 + beta3 + delta - eta - alpha (mod 2).
    RSSTensor<T> delta_share(x.shape), alpha_share(x.shape);
#pragma omp parallel for if (utils::should_parallelize(size))
    for (int i = 0; i < static_cast<int>(size); ++i)
    {
        if (party.party_id == 0)
        {
            delta_share.first.data[i] = delta_public.data[i];
            delta_share.second.data[i] = 0;
        }
        else if (party.party_id == 1)
        {
            delta_share.first.data[i] = 0;
            delta_share.second.data[i] = 0;
        }
        else
        {
            delta_share.first.data[i] = 0;
            delta_share.second.data[i] = delta_public.data[i];
        }

        alpha_share.first.data[i] = parameter.pc_cmp.alpha_with_pre;
        alpha_share.second.data[i] = parameter.pc_cmp.alpha_with_next;
    }

    RSSTensor<T> t1(x.shape), t2(x.shape);
    xor_share(beta_sum, delta_share, t1);
    xor_share(t1, eta, t2);
    xor_share(t2, alpha_share, theta);
}

template <typename T>
void rss_protocols::pc_msb(RSSTensor<T> &x, RSSTensor<T> &res,
                           Parameters<T> &parameter, const uint32_t size, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    const int msb_shift = sizeof(T) * 8 - 1;
    const T one = static_cast<T>(1);

    // term = MSB(a1) xor MSB(a2) xor MSB(a3) in RSS form.
    RSSTensor<T> msb_share_sum(x.shape);
#pragma omp parallel for if (utils::should_parallelize(size))
    for (int i = 0; i < static_cast<int>(size); i++)
    {
        msb_share_sum.first.data[i] = (x.first.data[i] >> msb_shift) & 1;
        msb_share_sum.second.data[i] = (x.second.data[i] >> msb_shift) & 1;
    }

    // theta2 = wrap3(2a1, 2a2, 2a3, L).
    RSSTensor<T> doubled_x(x.shape), theta2(x.shape);
    mulConst(x, static_cast<T>(2), doubled_x);
    utils::wrap3(doubled_x, theta2, parameter, malicious);

    // msb_bool = term xor theta2 (boolean-shared bit in RSS layout).
    RSSTensor<T> tmp1(x.shape), tmp2(x.shape), tmp_mul(x.shape), msb_bool(x.shape);
    add(msb_share_sum, theta2, tmp1);
    mul(msb_share_sum, theta2, tmp_mul);
    mulConst(tmp_mul, static_cast<T>(2), tmp2);
    sub(tmp1, tmp2, msb_bool);

    // ABY3 B2A: boolean shares (x1,x2,x3) -> arithmetic shares.
    RSSTensor<T> x1_a(x.shape), x2_a(x.shape), x3_a(x.shape);
    x1_a.zeros();
    x2_a.zeros();
    x3_a.zeros();

#pragma omp parallel for if (utils::should_parallelize(size))
    for (int i = 0; i < static_cast<int>(size); i++)
    {
        const T b_first = msb_bool.first.data[i] & one;
        const T b_second = msb_bool.second.data[i] & one;

        // RSS mapping:
        // P0 holds (x1, x2), P1 holds (x2, x3), P2 holds (x3, x1)
        if (party.party_id == 0)
        {
            x1_a.first.data[i] = b_first;
            x2_a.second.data[i] = b_second;
        }
        else if (party.party_id == 1)
        {
            x2_a.first.data[i] = b_first;
            x3_a.second.data[i] = b_second;
        }
        else
        {
            x3_a.first.data[i] = b_first;
            x1_a.second.data[i] = b_second;
        }
    }

    auto arith_xor = [&](RSSTensor<T> &a, RSSTensor<T> &b, RSSTensor<T> &out)
    {
        RSSTensor<T> ab(x.shape), two_ab(x.shape), sum_ab(x.shape);
        mul(a, b, ab, false, malicious);
        mulConst(ab, static_cast<T>(2), two_ab);
        add(a, b, sum_ab);
        sub(sum_ab, two_ab, out);
    };

    RSSTensor<T> d(x.shape);
    arith_xor(x1_a, x2_a, d);
    arith_xor(d, x3_a, res);
}

template <typename T>
void rss_protocols::nonNegative(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool isFloat, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    uint32_t size = x.size();
    RSSTensor<T> msb_res(x.shape);
    Tensor<T> mac_res_with_pre(x.shape), mac_res_with_next(x.shape);

    pc_msb(x, msb_res, parameter, size, malicious);
    // nonNegative = 1 - MSB
    sub((T)1, msb_res, res);
    if (isFloat)
    {
        mulConst(res, (T)(1 << x.float_scale_bit), res);
    }
}

template <typename T>
void rss_protocols::greaterEqual(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, Parameters<T> &parameter, bool isFloat, bool malicious)
{
    RSSTensor<T> z(x.shape);
    sub(x, y, z);
    nonNegative(z, res, parameter, isFloat, malicious);
}

template <typename T>
void rss_protocols::select(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, Parameters<T> &parameter, bool malicious)
{
     // 1. Get unscaled mask (0 or 1)
    // isFloat = false ensures we get pure 0/1 shares, not scaled by 2^f
    RSSTensor<T> mask(x.shape);
    nonNegative(x, mask, parameter, false, malicious);
    
    // 2. Multiply y by mask
    // Since mask is unscaled (0/1), the result has the same scale as y.
    // No truncation is needed (needTrunc = false).
    mul(y, mask, res, false, malicious);
}

template <typename T>
void rss_protocols::select(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, uint32_t y_num, Parameters<T> &parameter, bool malicious)
{
    uint32_t size = x.size();
    RSSTensor<T> mask(x.shape);
    // Get unscaled mask (0 or 1)
    nonNegative(x, mask, parameter, false, malicious);

    RSSTensor<T> expanded_mask(y.shape);
#pragma omp parallel for if (utils::should_parallelize(static_cast<uint64_t>(size) * y_num))
    for(uint32_t i = 0; i < y_num; ++i) {
        for(uint32_t j = 0; j < size; ++j) {
            expanded_mask.first.data[i * size + j] = mask.first.data[j];
            expanded_mask.second.data[i * size + j] = mask.second.data[j];
        }
    }
    // Multiply y by expanded mask (unscaled). No truncation needed.
    mul(y, expanded_mask, res, false, malicious);
}

template <typename T>
void rss_protocols::inv(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
    const T scale = static_cast<T>(1) << x.float_scale_bit;
    const T one_scaled = scale;
    const T two_scaled = static_cast<T>(2) * scale;
    const T w0_const = static_cast<T>(2.9142L * static_cast<long double>(scale));

    // Step 1: Bounding power alpha where 2^alpha <= b < 2^(alpha+1).
    // We compute alpha+1 via secure comparisons against public powers of 2.
    const int max_alpha = (static_cast<int>(8 * sizeof(T)) - x.float_scale_bit - 2 > 0)
                              ? (static_cast<int>(8 * sizeof(T)) - x.float_scale_bit - 2)
                              : 0;
    const uint32_t alpha_candidates = static_cast<uint32_t>(max_alpha + 1);
    const uint32_t size = x.size();

    RSSTensor<T> x_expand({size, alpha_candidates}), delta({size, alpha_candidates});
    Tensor<T> thresholds({size, alpha_candidates});

#pragma omp parallel for if (utils::should_parallelize(static_cast<uint64_t>(size) * alpha_candidates))
    for (int i = 0; i < static_cast<int>(size); ++i)
    {
        for (uint32_t j = 0; j < alpha_candidates; ++j)
        {
            const uint32_t idx = static_cast<uint32_t>(i) * alpha_candidates + j;
            x_expand.first.data[idx] = x.first.data[i];
            x_expand.second.data[idx] = x.second.data[i];
            thresholds.data[idx] = static_cast<T>((static_cast<uint64_t>(1) << j) * static_cast<uint64_t>(scale));
        }
    }

    sub(x_expand, thresholds, delta);
    nonNegative(delta, delta, parameters, false, malicious);

    RSSTensor<T> alpha_plus_one(x.shape);
    alpha_plus_one.zeros();
#pragma omp parallel for if (utils::should_parallelize(static_cast<uint64_t>(size) * alpha_candidates))
    for (int i = 0; i < static_cast<int>(size); ++i)
    {
        T sum_first = 0;
        T sum_second = 0;
        for (uint32_t j = 0; j < alpha_candidates; ++j)
        {
            const uint32_t idx = static_cast<uint32_t>(i) * alpha_candidates + j;
            sum_first += delta.first.data[idx];
            sum_second += delta.second.data[idx];
        }
        alpha_plus_one.first.data[i] = sum_first;
        alpha_plus_one.second.data[i] = sum_second;
    }

    Tensor<T> alpha_plus_one_open(x.shape);
    restore(alpha_plus_one, alpha_plus_one_open, malicious);

    // Step 2: Normalize b to x_norm in [0.5, 1) via x_norm = b / 2^(alpha+1).
    Tensor<T> inv_pow2_scaled(x.shape);
#pragma omp parallel for if (utils::should_parallelize(size))
    for (int i = 0; i < static_cast<int>(size); ++i)
    {
        uint32_t shift = static_cast<uint32_t>(alpha_plus_one_open.data[i]);
        if (shift > static_cast<uint32_t>(max_alpha + 1))
        {
            shift = static_cast<uint32_t>(max_alpha + 1);
        }
        const long double factor = std::ldexp(1.0L, -static_cast<int>(shift));
        inv_pow2_scaled.data[i] = static_cast<T>(factor * static_cast<long double>(scale));
        if (inv_pow2_scaled.data[i] == 0)
        {
            inv_pow2_scaled.data[i] = 1;
        }
    }

    RSSTensor<T> x_norm_tmp(x.shape), x_norm(x.shape);
    mulConst(x, inv_pow2_scaled, x_norm_tmp);
    truncate(x_norm_tmp, x_norm, scale, malicious);

    // Step 3: Initial approximation w0 = 2.9142 - 2 * x_norm.
    RSSTensor<T> two_x_norm(x.shape), w0(x.shape);
    mulConst(x_norm, static_cast<T>(2), two_x_norm);
    sub(w0_const, two_x_norm, w0);

    // Step 4: Error refinement with eps0, eps1.
    RSSTensor<T> xw0(x.shape), eps0(x.shape), eps1(x.shape);
    mul(x_norm, w0, xw0, true, malicious);
    sub(one_scaled, xw0, eps0);
    square(eps0, eps1, true, malicious);

    RSSTensor<T> one_plus_eps0(x.shape), one_plus_eps1(x.shape);
    add(eps0, one_scaled, one_plus_eps0);
    add(eps1, one_scaled, one_plus_eps1);

    RSSTensor<T> app_inv_x_norm(x.shape);
    mul(w0, one_plus_eps0, app_inv_x_norm, true, malicious);
    mul(app_inv_x_norm, one_plus_eps1, app_inv_x_norm, true, malicious);

    // Step 5: 1/b = (1/x_norm) / 2^(alpha+1).
    RSSTensor<T> inv_tmp(x.shape);
    mulConst(app_inv_x_norm, inv_pow2_scaled, inv_tmp);
    truncate(inv_tmp, res, scale, malicious);
}

template <typename T>
void rss_protocols::rsqrt(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
    (void)parameters;
    const T scale = static_cast<T>(1) << x.float_scale_bit;
    const T inv_sqrt2_scaled = static_cast<T>(0.70710678L * static_cast<long double>(scale));

    // Algorithm 1 (Separation): find shift = e + 1 so that x = u * 2^(e+1), u in [0.25, 0.5).
    // If 2^alpha <= x < 2^(alpha+1), then choose shift = alpha + 2.
    const int max_alpha = (static_cast<int>(8 * sizeof(T)) - x.float_scale_bit - 2 > 0)
                              ? (static_cast<int>(8 * sizeof(T)) - x.float_scale_bit - 2)
                              : 0;
    const uint32_t alpha_candidates = static_cast<uint32_t>(max_alpha + 1);
    const uint32_t size = x.size();

    RSSTensor<T> x_expand({size, alpha_candidates}), ge_bits({size, alpha_candidates});
    Tensor<T> thresholds({size, alpha_candidates});

#pragma omp parallel for if (utils::should_parallelize(static_cast<uint64_t>(size) * alpha_candidates))
    for (int i = 0; i < static_cast<int>(size); ++i)
    {
        for (uint32_t j = 0; j < alpha_candidates; ++j)
        {
            const uint32_t idx = static_cast<uint32_t>(i) * alpha_candidates + j;
            x_expand.first.data[idx] = x.first.data[i];
            x_expand.second.data[idx] = x.second.data[i];
            thresholds.data[idx] = static_cast<T>((static_cast<uint64_t>(1) << j) * static_cast<uint64_t>(scale));
        }
    }

    sub(x_expand, thresholds, ge_bits);
    nonNegative(ge_bits, ge_bits, parameters, false, malicious);

    RSSTensor<T> alpha_plus_one(x.shape);
    alpha_plus_one.zeros();
#pragma omp parallel for if (utils::should_parallelize(static_cast<uint64_t>(size) * alpha_candidates))
    for (int i = 0; i < static_cast<int>(size); ++i)
    {
        T sum_first = 0;
        T sum_second = 0;
        for (uint32_t j = 0; j < alpha_candidates; ++j)
        {
            const uint32_t idx = static_cast<uint32_t>(i) * alpha_candidates + j;
            sum_first += ge_bits.first.data[idx];
            sum_second += ge_bits.second.data[idx];
        }
        alpha_plus_one.first.data[i] = sum_first;
        alpha_plus_one.second.data[i] = sum_second;
    }

    Tensor<T> alpha_plus_one_open(x.shape);
    restore(alpha_plus_one, alpha_plus_one_open, malicious);

    Tensor<T> u_scale_factor(x.shape), comp_factor(x.shape);
#pragma omp parallel for if (utils::should_parallelize(size))
    for (int i = 0; i < static_cast<int>(size); ++i)
    {
        uint32_t alpha1 = static_cast<uint32_t>(alpha_plus_one_open.data[i]);
        if (alpha1 > static_cast<uint32_t>(max_alpha + 1))
        {
            alpha1 = static_cast<uint32_t>(max_alpha + 1);
        }

        const uint32_t shift = alpha1 + 1; // shift = e + 1

        const long double u_factor = std::ldexp(1.0L, -static_cast<int>(shift));
        u_scale_factor.data[i] = static_cast<T>(u_factor * static_cast<long double>(scale));
        if (u_scale_factor.data[i] == 0)
        {
            u_scale_factor.data[i] = 1;
        }

        const uint32_t half_shift = shift / 2;
        long double comp = std::ldexp(1.0L, -static_cast<int>(half_shift));
        if (shift & 1u)
        {
            comp *= static_cast<long double>(inv_sqrt2_scaled) / static_cast<long double>(scale);
        }
        comp_factor.data[i] = static_cast<T>(comp * static_cast<long double>(scale));
        if (comp_factor.data[i] == 0)
        {
            comp_factor.data[i] = 1;
        }
    }

    RSSTensor<T> u_tmp(x.shape), u(x.shape);
    mulConst(x, u_scale_factor, u_tmp);
    truncate(u_tmp, u, scale, malicious);

    // Algorithm 3 polynomial on [0.25, 0.5):
    // g(u) = 4.63887*u^2 - 5.77789*u + 3.14736
    // Horner form: c = 3.14736 + u * (4.63887*u - 5.77789)
    const T a_scaled = static_cast<T>(4.63887L * static_cast<long double>(scale));
    const T b_scaled = static_cast<T>(5.77789L * static_cast<long double>(scale));
    const T c_scaled = static_cast<T>(3.14736L * static_cast<long double>(scale));

    RSSTensor<T> a_mul_u(x.shape), a_mul_u_scaled(x.shape), inner(x.shape), poly_term(x.shape), c_approx(x.shape);
    mulConst(u, a_scaled, a_mul_u);
    truncate(a_mul_u, a_mul_u_scaled, scale, malicious);
    sub(a_mul_u_scaled, b_scaled, inner);
    mul(u, inner, poly_term, true, malicious);
    add(poly_term, c_scaled, c_approx);

    // Algorithm 2 (SqrtComp): y = c * 2^{-(e+1)/2}
    RSSTensor<T> y_tmp(x.shape);
    mulConst(c_approx, comp_factor, y_tmp);
    truncate(y_tmp, res, scale, malicious);
}

template <typename T>
void rss_protocols::gelu(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
    const T scale = static_cast<T>(1) << x.float_scale_bit;

    // Interval boundaries: (-inf,-4), [-4,-1.95), [-1.95,3], (3,+inf)
    const T c_neg4 = static_cast<T>(-4.0L * static_cast<long double>(scale));
    const T c_neg195 = static_cast<T>(-1.95L * static_cast<long double>(scale));
    const T c_pos3 = static_cast<T>(3.0L * static_cast<long double>(scale));
    const T c_pos3_eps = static_cast<T>(c_pos3 + 1); // strict 3 < x approximation

    // F0(x) on [-4,-1.95): c0 + c1*x + c2*x^2 + c3*x^3
    const T f0_c0 = static_cast<T>(-0.5054031199708174L * static_cast<long double>(scale));
    const T f0_c1 = static_cast<T>(-0.42226581151983866L * static_cast<long double>(scale));
    const T f0_c2 = static_cast<T>(-0.11807612951181953L * static_cast<long double>(scale));
    const T f0_c3 = static_cast<T>(-0.011034134030615728L * static_cast<long double>(scale));

    // F1(x) on [-1.95,3]: c0 + c1*x + c2*x^2 + c4*x^4 + c6*x^6
    const T f1_c0 = static_cast<T>(0.008526321541038084L * static_cast<long double>(scale));
    const T f1_c1 = static_cast<T>(0.5L * static_cast<long double>(scale));
    const T f1_c2 = static_cast<T>(0.3603292692789629L * static_cast<long double>(scale));
    const T f1_c4 = static_cast<T>(-0.037688200365904236L * static_cast<long double>(scale));
    const T f1_c6 = static_cast<T>(0.0018067462606141187L * static_cast<long double>(scale));

    // Step 1: secure interval indicators.
    RSSTensor<T> ge_neg4(x.shape), ge_neg195(x.shape), gt_pos3(x.shape);
    RSSTensor<T> tmp(x.shape), one_minus_b0(x.shape), one_minus_b2(x.shape);
    RSSTensor<T> b0(x.shape), b1(x.shape), b2(x.shape);
    RSSTensor<T> z0(x.shape), z1(x.shape), z2(x.shape);

    add(x, static_cast<T>(4.0L * static_cast<long double>(scale)), tmp);
    nonNegative(tmp, ge_neg4, parameters, false, malicious); // x >= -4
    add(x, static_cast<T>(1.95L * static_cast<long double>(scale)), tmp);
    nonNegative(tmp, ge_neg195, parameters, false, malicious); // x >= -1.95
    sub(x, c_pos3_eps, tmp);
    nonNegative(tmp, gt_pos3, parameters, false, malicious); // x > 3

    // b0 = (x < -4), b1 = (x < -1.95), b2 = (3 < x)
    sub(static_cast<T>(1), ge_neg4, b0);
    sub(static_cast<T>(1), ge_neg195, b1);
    b2 = gt_pos3;

    // z0: [-4, -1.95), z1: [-1.95, 3], z2: (3, +inf)
    sub(static_cast<T>(1), b0, one_minus_b0);
    mul(one_minus_b0, b1, z0, false, malicious);

    sub(static_cast<T>(1), b2, one_minus_b2);
    mul(ge_neg195, one_minus_b2, z1, false, malicious);

    z2 = b2;

    // Step 2: powers x^2, x^3, x^4, x^6.
    RSSTensor<T> x2(x.shape), x3(x.shape), x4(x.shape), x6(x.shape);
    square(x, x2, true, malicious);
    mul(x2, x, x3, true, malicious);
    square(x2, x4, true, malicious);
    mul(x4, x2, x6, true, malicious);

    auto eval_poly_f0 = [&](T c0, T c1, T c2, T c3, RSSTensor<T> &out)
    {
        RSSTensor<T> term(x.shape), acc(x.shape);
        acc.zeros();
        add(acc, c0, acc);

        mulConst(x, c1, term);
        truncate(term, term, scale, malicious);
        add(acc, term, acc);

        mulConst(x2, c2, term);
        truncate(term, term, scale, malicious);
        add(acc, term, acc);

        mulConst(x3, c3, term);
        truncate(term, term, scale, malicious);
        add(acc, term, acc);

        out = acc;
    };

    auto eval_poly_f1 = [&](T c0, T c1, T c2, T c4, T c6, RSSTensor<T> &out)
    {
        RSSTensor<T> term(x.shape), acc(x.shape);
        acc.zeros();
        add(acc, c0, acc);

        mulConst(x, c1, term);
        truncate(term, term, scale, malicious);
        add(acc, term, acc);

        mulConst(x2, c2, term);
        truncate(term, term, scale, malicious);
        add(acc, term, acc);

        mulConst(x4, c4, term);
        truncate(term, term, scale, malicious);
        add(acc, term, acc);

        mulConst(x6, c6, term);
        truncate(term, term, scale, malicious);
        add(acc, term, acc);

        out = acc;
    };

    RSSTensor<T> f0(x.shape), f1(x.shape);
    eval_poly_f0(f0_c0, f0_c1, f0_c2, f0_c3, f0);
    eval_poly_f1(f1_c0, f1_c1, f1_c2, f1_c4, f1_c6, f1);

    // Step 3/4: secure multiplexer
    // y = z0*F0(x) + z1*F1(x) + z2*x
    RSSTensor<T> y0(x.shape), y1(x.shape), y2(x.shape), y01(x.shape);
    mul(f0, z0, y0, false, malicious);
    mul(f1, z1, y1, false, malicious);
    mul(x, z2, y2, false, malicious);
    add(y0, y1, y01);
    add(y01, y2, res);
}

template <typename T>
void rss_protocols::silu(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
    const T scale = static_cast<T>(1) << x.float_scale_bit;
    const T c_neg8 = static_cast<T>(-8.0L * static_cast<long double>(scale));
    const T c_neg4 = static_cast<T>(-4.0L * static_cast<long double>(scale));
    const T c_pos4 = static_cast<T>(4.0L * static_cast<long double>(scale));
    const T c_pos4_eps = static_cast<T>(c_pos4 + 1); // strict x > 4

    // Coefficients from flax_llama7b.py hack_silu.
    // seg1(x) on [-8,-4): a2*x^2 + a1*x + a0
    const T a0 = static_cast<T>(-0.3067541139982155L * static_cast<long double>(scale));
    const T a1 = static_cast<T>(-0.0819767021525476L * static_cast<long double>(scale));
    const T a2 = static_cast<T>(-0.0055465625580307L * static_cast<long double>(scale));

    // seg2(x) on [-4,4): b6*x^6 + b4*x^4 + b2*x^2 + b1*x + b0
    const T b0 = static_cast<T>(0.0085064025895951L * static_cast<long double>(scale));
    const T b1 = static_cast<T>(0.5L * static_cast<long double>(scale));
    const T b2 = static_cast<T>(0.2281430841728270L * static_cast<long double>(scale));
    const T b4 = static_cast<T>(-0.011113046708173L * static_cast<long double>(scale));
    const T b6 = static_cast<T>(0.0002743776353465L * static_cast<long double>(scale));

    auto xor01 = [&](RSSTensor<T> &lhs, RSSTensor<T> &rhs, RSSTensor<T> &out)
    {
        RSSTensor<T> t1(x.shape), t2(x.shape), t3(x.shape);
        add(lhs, rhs, t1);
        mul(lhs, rhs, t2, false, malicious);
        mulConst(t2, static_cast<T>(2), t3);
        sub(t1, t3, out);
    };

    // b0 = [x < -8], b1 = [x < -4], b2 = [x > 4]
    RSSTensor<T> ge_neg8(x.shape), ge_neg4(x.shape), gt_pos4(x.shape);
    RSSTensor<T> tmp(x.shape), m0(x.shape), m1(x.shape), m2(x.shape), one_const(x.shape);
    add(x, static_cast<T>(8.0L * static_cast<long double>(scale)), tmp);
    nonNegative(tmp, ge_neg8, parameters, false, malicious);
    add(x, static_cast<T>(4.0L * static_cast<long double>(scale)), tmp);
    nonNegative(tmp, ge_neg4, parameters, false, malicious);
    sub(x, c_pos4_eps, tmp);
    nonNegative(tmp, gt_pos4, parameters, false, malicious);

    sub(static_cast<T>(1), ge_neg8, m0); // x < -8
    sub(static_cast<T>(1), ge_neg4, m1); // x < -4
    m2 = gt_pos4;                        // x > 4

    // b3 = x in [-4,4) = b1 xor b2 xor 1
    // b4 = x in [-8,-4) = b0 xor b1
    RSSTensor<T> b3(x.shape), b4m(x.shape), b3_tmp(x.shape);
    xor01(m1, m2, b3_tmp);
    sub(static_cast<T>(1), b3_tmp, b3);
    xor01(m0, m1, b4m);

    RSSTensor<T> x2(x.shape), x4(x.shape), x6(x.shape);
    square(x, x2, true, malicious);
    square(x2, x4, true, malicious);
    mul(x2, x4, x6, true, malicious);

    // seg1 = a2*x^2 + a1*x + a0
    RSSTensor<T> seg1(x.shape), term(x.shape), acc(x.shape);
    acc.zeros();
    add(acc, a0, acc);
    mulConst(x, a1, term);
    truncate(term, term, scale, malicious);
    add(acc, term, acc);
    mulConst(x2, a2, term);
    truncate(term, term, scale, malicious);
    add(acc, term, acc);
    seg1 = acc;

    // seg2 = b6*x^6 + b4*x^4 + b2*x^2 + b1*x + b0
    RSSTensor<T> seg2(x.shape);
    acc.zeros();
    add(acc, b0, acc);
    mulConst(x, b1, term);
    truncate(term, term, scale, malicious);
    add(acc, term, acc);
    mulConst(x2, b2, term);
    truncate(term, term, scale, malicious);
    add(acc, term, acc);
    mulConst(x4, b4, term);
    truncate(term, term, scale, malicious);
    add(acc, term, acc);
    mulConst(x6, b6, term);
    truncate(term, term, scale, malicious);
    add(acc, term, acc);
    seg2 = acc;

    // ret = (x > 4) * x + (x in [-8,-4)) * seg1 + (x in [-4,4)) * seg2
    RSSTensor<T> y0(x.shape), y1(x.shape), y2(x.shape), y01(x.shape);
    mul(x, m2, y0, false, malicious);
    mul(seg1, b4m, y1, false, malicious);
    mul(seg2, b3, y2, false, malicious);
    add(y0, y1, y01);
    add(y01, y2, res);
}

template <typename T>
void rss_protocols::rms_norm(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious, long double eps)
{
    const T scale = static_cast<T>(1) << x.float_scale_bit;
    const T eps_scaled = static_cast<T>(eps * static_cast<long double>(scale));
    const uint32_t dim_size = x.shape.back();
    const uint32_t common_size = x.size() / dim_size;

    std::vector<uint32_t> reduce_shape = x.shape;
    reduce_shape.pop_back();

    RSSTensor<T> x2(x.shape);
    square(x, x2, true, malicious);

    RSSTensor<T> mean_sq(reduce_shape);
#pragma omp parallel for if (utils::should_parallelize(static_cast<uint64_t>(common_size) * dim_size))
    for (int i = 0; i < static_cast<int>(common_size); ++i)
    {
        mean_sq.first.data[i] = 0;
        mean_sq.second.data[i] = 0;
        for (uint32_t j = 0; j < dim_size; ++j)
        {
            const uint32_t idx = static_cast<uint32_t>(i) * dim_size + j;
            mean_sq.first.data[i] += x2.first.data[idx];
            mean_sq.second.data[i] += x2.second.data[idx];
        }
    }
    truncate(mean_sq, dim_size, malicious);
    add(mean_sq, eps_scaled, mean_sq);

    RSSTensor<T> inv_rms(reduce_shape);
    rsqrt(mean_sq, inv_rms, parameters, malicious);

    RSSTensor<T> inv_rms_broadcast(x.shape);
#pragma omp parallel for if (utils::should_parallelize(static_cast<uint64_t>(common_size) * dim_size))
    for (int i = 0; i < static_cast<int>(common_size); ++i)
    {
        for (uint32_t j = 0; j < dim_size; ++j)
        {
            const uint32_t idx = static_cast<uint32_t>(i) * dim_size + j;
            inv_rms_broadcast.first.data[idx] = inv_rms.first.data[i];
            inv_rms_broadcast.second.data[idx] = inv_rms.second.data[i];
        }
    }

    mul(x, inv_rms_broadcast, res, true, malicious);
}

template <typename T>
void rss_protocols::max_last_dim(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool malicious)
{
    uint32_t last_dim_size = x.shape.back();
    RSSTensor<T> tmp(x), even, odd, delta;
    uint32_t freeze_size = res.size();
    std::vector<uint32_t> new_shape = res.shape;
    uint32_t half_size;

    int index0, index1;

    while (last_dim_size > 1)
    {
        half_size = (last_dim_size + 1) / 2;
        new_shape.push_back(half_size);

        even.allocate(new_shape);
        odd.allocate(new_shape);
        delta.allocate(new_shape);

        for (uint32_t i = 0; i < freeze_size; ++i)
        {
            for (uint32_t j = 0; j < half_size - 1; ++j)
            {
                index0 = i * half_size + j;
                index1 = i * last_dim_size + 2 * j;
                even.first.data[index0] = tmp.first.data[index1];
                even.second.data[index0] = tmp.second.data[index1];

                odd.first.data[index0] = tmp.first.data[index1 + 1];
                odd.second.data[index0] = tmp.second.data[index1 + 1];
            }
            index0 = i * half_size + half_size - 1;
            index1 = i * last_dim_size + 2 * (half_size - 1);
            even.first.data[index0] = tmp.first.data[index1];
            even.second.data[index0] = tmp.second.data[index1];

            odd.first.data[index0] = tmp.first.data[i * last_dim_size + last_dim_size - 1];
            odd.second.data[index0] = tmp.second.data[i * last_dim_size + last_dim_size - 1];
        }

        sub(even, odd, delta);
        select(delta, delta, delta, parameter, malicious);
        add(odd, delta, even);
        tmp = even;
        even.free();
        odd.free();
        last_dim_size = half_size;
        new_shape = res.shape;
    }
    res = tmp;
}

template <typename T>
void rss_protocols::neg_exp(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool malicious)
{
    const T scale = static_cast<T>(1) << x.float_scale_bit;
    const T t_exp = static_cast<T>(-14) * scale;
    constexpr int t = 5; // neg_Exp(x) = (1 + x/2^t)^(2^t)

    // mask = [x >= T_exp], values below threshold are set to zero.
    RSSTensor<T> x_minus_t(x.shape), keep_mask(x.shape);
    sub(x, t_exp, x_minus_t);
    nonNegative(x_minus_t, keep_mask, parameter, false, malicious);

    // base = 1 + x / 2^t
    RSSTensor<T> x_div_pow2t(x.shape), base(x.shape);
    truncate(x, x_div_pow2t, static_cast<T>(1) << t, malicious);
    add(x_div_pow2t, scale, base);

    // Repeated squaring t times to realize power 2^t.
    RSSTensor<T> exp_approx(base.shape);
    exp_approx = base;
    for (int i = 0; i < t; ++i)
    {
        square(exp_approx, exp_approx, true, malicious);
    }

    // Apply threshold mask (0/1, unscaled), keep scale of exp_approx unchanged.
    mul(exp_approx, keep_mask, res, false, malicious);
}

template <typename T>
void rss_protocols::softmax_forward(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool malicious)
{
    /* only support dim = -1*/
    std::vector<uint32_t> sum_shape = x.shape;
    uint32_t dim_size = sum_shape.back();
    sum_shape.pop_back();

    RSSTensor<T> x_max(sum_shape), delta(x.shape);
    uint32_t common_size = x_max.size();
    max_last_dim(x, x_max, parameter, malicious);
    int index;
#pragma omp parallel for if (utils::should_parallelize(static_cast<uint64_t>(common_size) * dim_size))
    for (int i = 0; i < common_size; i++)
    {
        for (int j = 0; j < dim_size; j++)
        {
            index = i * dim_size + j;
            delta.first.data[index] = x.first.data[index] - x_max.first.data[i];
            delta.second.data[index] = x.second.data[index] - x_max.second.data[i];
        }
    }

    // Enforce strictly negative exponent input: (x - max) - epsilon.
    sub(delta, static_cast<T>(1), delta);

    RSSTensor<T> exp_x(x.shape);
    neg_exp(delta, exp_x, parameter, malicious);
    RSSTensor<T> sum(sum_shape);

#pragma omp parallel for if (utils::should_parallelize(static_cast<uint64_t>(common_size) * dim_size))
    for (int i = 0; i < common_size; i++)
    {
        sum.first.data[i] = 0;
        sum.second.data[i] = 0;
        for (int j = 0; j < dim_size; j++)
        {
            sum.first.data[i] += exp_x.first.data[i * dim_size + j];
            sum.second.data[i] += exp_x.second.data[i * dim_size + j];
        }
    }

    inv(sum, sum, parameter, malicious);
    RSSTensor<T> broadcast_sum(x.shape);
#pragma omp parallel for if (utils::should_parallelize(static_cast<uint64_t>(common_size) * dim_size))
    for (int i = 0; i < common_size; i++)
    {
        for (int j = 0; j < dim_size; j++)
        {
            broadcast_sum.first.data[i * dim_size + j] = sum.first.data[i];
            broadcast_sum.second.data[i * dim_size + j] = sum.second.data[i];
        }
    }
    mul(exp_x, broadcast_sum, res, true, malicious);
}

template <typename T>
void rss_protocols::grouped_multi_query_attention(RSSTensor<T> &query, RSSTensor<T> &key, RSSTensor<T> &value, RSSTensor<T> &res,
                                                  Parameters<T> &parameter, uint32_t num_heads,
                                                  uint32_t num_key_value_heads, bool masked, bool malicious)
{
    always_assert(query.shape.size() == 3 && key.shape.size() == 3 && value.shape.size() == 3);
    always_assert(query.shape[0] == key.shape[0] && key.shape[0] == value.shape[0]);
    always_assert(query.shape[1] == key.shape[1] && key.shape[1] == value.shape[1]);

    const uint32_t batch_size = query.shape[0];
    const uint32_t seq_len = query.shape[1];
    const uint32_t hidden_size = query.shape[2];
    always_assert(num_heads > 0 && num_key_value_heads > 0);
    always_assert(hidden_size % num_heads == 0);

    const uint32_t head_size = hidden_size / num_heads;
    const uint32_t key_value_size = key.shape[2];
    always_assert(key_value_size == value.shape[2]);
    always_assert(key_value_size == num_key_value_heads * head_size);
    always_assert(num_heads % num_key_value_heads == 0);

    RSSTensor<T> query_heads = query;
    query_heads.reshape({batch_size, seq_len, num_heads, head_size});
    RSSTensor<T> key_heads = key;
    key_heads.reshape({batch_size, seq_len, num_key_value_heads, head_size});
    RSSTensor<T> value_heads = value;
    value_heads.reshape({batch_size, seq_len, num_key_value_heads, head_size});

    RSSTensor<T> query_layer = query_heads.permute({0, 2, 1, 3});
    RSSTensor<T> key_layer = key_heads.permute({0, 2, 3, 1});
    RSSTensor<T> value_layer = value_heads.permute({0, 2, 1, 3});

    RSSTensor<T> key_for_attn;
    RSSTensor<T> value_for_attn;
    if (num_key_value_heads == num_heads)
    {
        key_for_attn = key_layer;
        value_for_attn = value_layer;
    }
    else
    {
        key_for_attn.allocate({batch_size, num_heads, head_size, seq_len});
        value_for_attn.allocate({batch_size, num_heads, seq_len, head_size});
        const uint32_t group_size = num_heads / num_key_value_heads;

#pragma omp parallel for if (parallel::should_parallelize(static_cast<uint64_t>(batch_size) * num_heads * seq_len * head_size))
        for (int b = 0; b < static_cast<int>(batch_size); b++)
        {
            for (uint32_t h = 0; h < num_heads; h++)
            {
                const uint32_t kv_head = h / group_size;
                for (uint32_t s = 0; s < seq_len; s++)
                {
                    for (uint32_t d = 0; d < head_size; d++)
                    {
                        const uint32_t key_dst = ((static_cast<uint32_t>(b) * num_heads + h) * head_size + d) * seq_len + s;
                        const uint32_t key_src = ((static_cast<uint32_t>(b) * num_key_value_heads + kv_head) * head_size + d) * seq_len + s;
                        key_for_attn.first.data[key_dst] = key_layer.first.data[key_src];
                        key_for_attn.second.data[key_dst] = key_layer.second.data[key_src];

                        const uint32_t value_dst = ((static_cast<uint32_t>(b) * num_heads + h) * seq_len + s) * head_size + d;
                        const uint32_t value_src = ((static_cast<uint32_t>(b) * num_key_value_heads + kv_head) * seq_len + s) * head_size + d;
                        value_for_attn.first.data[value_dst] = value_layer.first.data[value_src];
                        value_for_attn.second.data[value_dst] = value_layer.second.data[value_src];
                    }
                }
            }
        }
    }

    RSSTensor<T> attention_scores({batch_size, num_heads, seq_len, seq_len});
    rss_protocols::matMul(query_layer, key_for_attn, attention_scores, true, malicious);
    rss_protocols::truncate(attention_scores, static_cast<uint32_t>(std::sqrt(static_cast<double>(head_size))), malicious);

    if (masked)
    {
        const uint32_t common_size = batch_size * num_heads;
#pragma omp parallel for if (parallel::should_parallelize(static_cast<uint64_t>(common_size) * seq_len * seq_len))
        for (int i = 0; i < static_cast<int>(common_size); i++)
        {
            for (uint32_t q = 0; q < seq_len; q++)
            {
                const uint32_t row_base = (static_cast<uint32_t>(i) * seq_len + q) * seq_len;
                const T fill_first = attention_scores.first.data[row_base];
                const T fill_second = attention_scores.second.data[row_base];
                for (uint32_t k = q + 1; k < seq_len; k++)
                {
                    attention_scores.first.data[row_base + k] = fill_first;
                    attention_scores.second.data[row_base + k] = fill_second;
                }
            }
        }
    }

    RSSTensor<T> attention_probs(attention_scores.shape);
    rss_protocols::softmax_forward(attention_scores, attention_probs, parameter, malicious);

    if (masked)
    {
        const uint32_t common_size = batch_size * num_heads;
#pragma omp parallel for if (parallel::should_parallelize(static_cast<uint64_t>(common_size) * seq_len * seq_len))
        for (int i = 0; i < static_cast<int>(common_size); i++)
        {
            for (uint32_t q = 0; q < seq_len; q++)
            {
                const uint32_t row_base = (static_cast<uint32_t>(i) * seq_len + q) * seq_len;
                for (uint32_t k = q + 1; k < seq_len; k++)
                {
                    attention_probs.first.data[row_base + k] = 0;
                    attention_probs.second.data[row_base + k] = 0;
                }
            }
        }
    }

    RSSTensor<T> context_layer({batch_size, num_heads, seq_len, head_size});
    rss_protocols::matMul(attention_probs, value_for_attn, context_layer, true, malicious);

    RSSTensor<T> attn_output = context_layer.permute({0, 2, 1, 3});
    attn_output.reshape({batch_size, seq_len, hidden_size});
    res = attn_output;
}

