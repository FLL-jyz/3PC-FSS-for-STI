#pragma once

#include <fstream>
#include <cmath>
#include "globals.h"
#include "params.h"
#include "rss_protocols.h"
#include "ass_protocols.h"
#include "party3pc.h"
#include "dcf.h"
#include "dpf.h"
#include "fss.h"

namespace rss_protocols
{
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
        void UCMP(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter);

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
    void lut(RSSTensor<T> &x, RSSTensor<T> &res, LUT_Param<T> &parameter, bool malicious = false);

    template <typename T>
    void lut(RSSTensor<T> &x, RSSTensor<T> &res1, RSSTensor<T> &res2, LUT_Param<T> &parameter1, LUT_Param<T> &parameter2, bool malicious = false);

    template <typename T>
    void inv(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious = false);

    template <typename T>
    void rsqrt(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious = false);

    template <typename T>
    void gelu(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious = false);

    template <typename T>
    void max_last_dim(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious = false);

    template <typename T>
    void neg_exp(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &Parameters, bool malicious = false);

    template <typename T>
    void softmax_forward(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool malicious = false);

    template <typename T, typename U>
    void downcast(RSSTensor<T> &x, RSSTensor<U> &res);

    template <typename U, typename T>
    void upcast(RSSTensor<U> &x, RSSTensor<T> &res, int party_id, bool malicious = false);
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
        for (uint32_t i = 0; i < broadcast_size; i++)
        {
            for (uint32_t j = 0; j < common_size; j++)
            {
#pragma omp parallel for
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
        for (uint32_t i = 0; i < broadcast_size; i++)
        {
            for (uint32_t j = 0; j < common_size; j++)
            {
#pragma omp parallel for
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
        for (uint32_t i = 0; i < broadcast_size; i++)
        {
            for (uint32_t j = 0; j < common_size; j++)
            {
#pragma omp parallel for
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
        for (uint32_t i = 0; i < broadcast_size; i++)
        {
            for (uint32_t j = 0; j < common_size; j++)
            {
#pragma omp parallel for
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
        for (uint32_t i = 0; i < broadcast_size; i++)
        {
            for (uint32_t j = 0; j < common_size; j++)
            {
#pragma omp parallel for
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
        for (uint32_t i = 0; i < broadcast_size; i++)
        {
            for (uint32_t j = 0; j < common_size; j++)
            {
#pragma omp parallel for
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

#pragma omp parallel for
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

#pragma omp parallel for
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
#pragma omp parallel for
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

#pragma omp parallel for
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
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] + y.data[i];
            res.second.data[i] = x.second.data[i];
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i];
        }
    }
    else
    {
#pragma omp parallel for
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
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] + y;
            res.second.data[i] = x.second.data[i];
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i];
        }
    }
    else
    {
#pragma omp parallel for
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

#pragma omp parallel for
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
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] - y.data[i];
            res.second.data[i] = x.second.data[i];
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i];
        }
    }
    else
    {
#pragma omp parallel for
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
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] - y;
            res.second.data[i] = x.second.data[i];
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i];
        }
    }
    else
    {
#pragma omp parallel for
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
#pragma omp parallel for
        for (int i = 0; i < y.size(); i++)
        {
            res.first.data[i] = x - y.first.data[i];
            res.second.data[i] = -y.second.data[i];
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < y.size(); i++)
        {
            res.first.data[i] = -y.first.data[i];
            res.second.data[i] = -y.second.data[i];
        }
    }
    else
    {
#pragma omp parallel for
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

#pragma omp parallel for
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

#pragma omp parallel for
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

        RSSTensor<T> zero_res({size});

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
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + a.second.data[i] * rho.data[i] * 2 -
                                          rho.data[i] * rho.data[i];
            }
        }

        std::thread t1([&](){ party.send_tensor_to(party.next_party_id, zero_res.first); });
        t1.join();
        Tensor<T> tmp_recv({size});
        std::thread t2([&](){ party.recv_tensor_from(party.pre_party_id, tmp_recv); });
        t2.join();

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
    int threads_num = omp_get_max_threads();
    omp_set_num_threads(64);

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

#pragma omp parallel for
        for (int i = 0; i < x_size; i++)
        {
            combined.first.data[i] = a.first.data[i] - x.first.data[i];
            combined.second.data[i] = a.second.data[i] - x.second.data[i];
        }
#pragma omp parallel for
        for (int i = 0; i < y_size; i++)
        {
            combined.first.data[i + x_size] = b.first.data[i] - y.first.data[i];
            combined.second.data[i + x_size] = b.second.data[i] - y.second.data[i];
        }

        Tensor<T> rhoSigma({combined_size}), rho(x.shape), sigma(y.shape);
        restore(combined, rhoSigma, malicious);

#pragma omp parallel for
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
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] =
                    res.first.data[i] - c.first.data[i] + af.first.data[i] + eb.first.data[i] - ef.data[i];
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + af.second.data[i] + eb.second.data[i];
            }
        }
        else if (party.party_id == 1)
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + af.first.data[i] + eb.first.data[i];
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + af.second.data[i] + eb.second.data[i];
            }
        }
        else
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + af.first.data[i] + eb.first.data[i];
                zero_res.second.data[i] =
                    res.second.data[i] - c.second.data[i] + af.second.data[i] + eb.second.data[i] - ef.data[i];
            }
        }

        party.send_tensor_to(party.next_party_id, zero_res.first);
        Tensor<T> tmp_recv(res.shape);
        party.recv_tensor_from(party.pre_party_id, tmp_recv);

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            always_assert(zero_res.first.data[i] + zero_res.second.data[i] + tmp_recv.data[i] == 0);
        }
    }
    omp_set_num_threads(threads_num);
    if (needTrunc)
    {   
        truncate(res, malicious);
    }
}

// template <typename T>
// void rss_protocols::truncate(RSSTensor<T> &x, RSSTensor<T> &res, size_t scale, bool malicious)
// {
//     Party3PC &party = Party3PC::getInstance();
//     uint32_t size = x.size();
//     uint32_t s = (uint32_t)log2(scale);
//     uint32_t bitlength = sizeof(T) * 8;
    
//     Parameters<T> parameter;
//     parameter.init_trunc(s);

//     // 1. Compute x_hat = x + r
//     RSSTensor<T> x_hat(x.shape);
//     RSSTensor<T> r(x.shape);
    
//     // Use stored r (index 0 for simulation)
//     for(uint32_t i=0; i<size; ++i) {
//         r.first.data[i] = parameter.trunc_param.r_share[party.party_id];
//         r.second.data[i] = parameter.trunc_param.r_share[(party.party_id + 1) % 3];
//     }
//     add(x, r, x_hat);
    
//     // 2. Open x_hat
//     Tensor<T> x_hat_open(x.shape);
//     restore(x_hat, x_hat_open, malicious);
    
//     // 3. sbitcmp = VDCF(x_hat, keys)
//     RSSTensor<T> sbitcmp(x.shape);
//     Tensor<T> temp_res(x.shape);
//     for(uint32_t i=0; i<size; ++i) {
//         T x_hat_val = x_hat_open.data[i];
//         T v_input = x_hat_val & ((1ULL << s) - 1);
        
//         GroupElement res_elem(0, 1);
//         GroupElement x_elem(v_input, bitlength);
        
//         evalDCF(party.party_id, &res_elem, x_elem, parameter.trunc_param.keys[party.party_id]);
//         temp_res.data[i] = res_elem.value;
//     }
//     reshare(temp_res, sbitcmp);
    
//     // 4. signcmp = nonNegative(x)
//     RSSTensor<T> signcmp(x.shape);
//     // Swap DReLU params
//     auto backup_drelu = parameter.DRelu;
//     parameter.DRelu = *parameter.trunc_param.drelu;
//     nonNegative(x, signcmp, parameter, size, malicious);
//     parameter.DRelu = backup_drelu;
    
//     // 5. Compute w
//     // w = [(b mod 2) * x_hat_shift + <y_shift> + <sbitcmp>] - 2^{l-s-1} * <signcmp>
//     // Note: x_hat_shift = x_hat >> s
//     // <y_shift> is in parameter.trunc_param.y_shift_share
    
//     RSSTensor<T> w(x.shape);
//     T two_pow_l_s_1 = (T)1 << (bitlength - s - 1);

//     for (uint32_t i = 0; i < size; ++i) {
//         T x_hat_shift = x_hat_open.data[i] >> s;
//         T y_shift_share0 = parameter.trunc_param.y_shift_share[party.party_id];
//         T y_shift_share1 = parameter.trunc_param.y_shift_share[(party.party_id + 1) % 3];

//         T w_first = 0;
//         T w_second = 0;

//         // <y_shift>
//         w_first += y_shift_share0;
//         w_second += y_shift_share1;

//         // <sbitcmp>
//         w_first += sbitcmp.first.data[i];
//         w_second += sbitcmp.second.data[i];

//         // - 2^{l-s-1} * <signcmp>
//         w_first -= two_pow_l_s_1 * signcmp.first.data[i];
//         w_second -= two_pow_l_s_1 * signcmp.second.data[i];

//         res.first.data[i] = w_first;
//         res.second.data[i] = w_second;
//         // (b mod 2) * x_hat_shift
//         // Only party 1 adds x_hat_shift according to the protocol
//         add(res, x_hat_shift, res);
//     }
// }

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
void rss_protocols::utils::UCMP(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter)
{   
    Party3PC &party = Party3PC::getInstance();
    uint32_t size = x.size();
    uint64_t bitlength = sizeof(T) * 8;
    
    // 1. Reconstruct x
    RSSTensor<T> x_hat_share(x.shape);
    for (uint32_t i = 0; i < x.size(); ++i) {
        x_hat_share.first.data[i] = x.first.data[i] + parameter.ucmp_param.r_in_share[party.party_id];
        int next_id = (party.party_id + 1) % 3;
        x_hat_share.second.data[i] = x.second.data[i] + parameter.ucmp_param.r_in_share[next_id];
    }
    Tensor<T> x_open(x.shape);
    restore(x_hat_share, x_open); 
    
    // 2. compute comparison
    res.assert_same_shape(x);
    Tensor<T> temp_res(x.shape);
#pragma omp parallel for
    for(uint32_t i=0; i<size; ++i) {
        T idx = x_open.data[i];
        GroupElement res_elem(0, 1);
        GroupElement x_elem(idx, bitlength);
        
        // Evaluate DCF
        // keys[i] is the key for the current party
        // evalDCF(party.party_id, &res_elem, x_elem, parameter.ucmp_param.keys[i]);
        // temp_res.data[i] = res_elem.value;
        temp_res.data[i] = x_open.data[i];
    }
    reshare(temp_res, res);
}

template <typename T>
void rss_protocols::pc_msb(RSSTensor<T> &x, RSSTensor<T> &res,
                           Parameters<T> &parameter, const uint32_t size, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    uint32_t bitlength = sizeof(T) * 8;

    // 1. Compute x_hat = x + r
    RSSTensor<T> x_hat(x.shape);
    // Add r (RSS) to x (RSS). r is stored in parameter.DRelu.r_share
#pragma omp parallel for
    for(int i=0; i<size; ++i) {
        x_hat.first.data[i] = x.first.data[i] + parameter.DRelu.r_in_share[party.party_id];
        x_hat.second.data[i] = x.second.data[i] + parameter.DRelu.r_in_share[(party.party_id+1) % 3];
    }

    // 2. Open x_hat
    Tensor<T> x_open(x.shape);
    restore(x_hat, x_open);

    // 3. Prepare UCMP input and Run UCMP
    // Input = 2^{l-1} - 1 - (x_hat mod 2^{l-1})
    T max_pos = (T(1) << (bitlength - 1)) - 1;
    
    RSSTensor<T> ucmp_res(x.shape);
    ucmp_res.assert_same_shape(x);
    Tensor<T> temp_res(x.shape);
    
#pragma omp parallel for
    for(int i=0; i<size; ++i) {
        T x_hat_val = x_open.data[i];
        T ucmp_input = x_hat_val & max_pos;
        
        GroupElement res_elem(0, 1);
        GroupElement x_elem(ucmp_input, bitlength);
        
        // Use stored DCF key (index 0 for simulation)
        // evalDCF(party.party_id, &res_elem, x_elem, parameter.DRelu.keys[party.party_id]);
        // temp_res.data[i] = res_elem.value;
        temp_res.data[i] = x_open.data[i];
    }
    reshare(temp_res, ucmp_res);

    // 4. Compute w = (b mod 2) * MSB(x_hat) + c + res
    RSSTensor<T> w(x.shape);
#pragma omp parallel for
    for(int i=0; i<size; ++i) {
        // Add c (RSS) + res (RSS)
        w.first.data[i] = ucmp_res.first.data[i] + parameter.DRelu.c_in_share[party.party_id];
        w.second.data[i] = ucmp_res.second.data[i] + parameter.DRelu.c_in_share[(party.party_id+1) % 3];

        // Add MSB(x_hat) (Public value)
        // MSB(x_hat) is 0 or 1
        T msb = x_open.data[i] >> (bitlength - 1);

        // Add public value to RSS shares:
        // P0: s0 + k, s1
        // P1: s1, s2
        // P2: s2, s0 + k
        if (party.party_id == 0)
        {
            w.first.data[i] += msb;
        }
        else if (party.party_id == 2)
        {
            w.second.data[i] += msb;
        }
    }
    // Output result
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        res.first.data[i] = w.first.data[i];
        res.second.data[i] = w.second.data[i];
    }
}

template <typename T>
void rss_protocols::nonNegative(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool isFloat, bool malicious)
{
    uint32_t size = x.size();
    // pc_msb returns the sign bit (MSB): 1 if x >= 0, 0 if x < 0
    pc_msb(x, res, parameter, size, malicious);
    // nonNegative = 1 - MSB
    sub((T)1, res, res);
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
#pragma omp parallel for
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
void rss_protocols::lut(RSSTensor<T> &x, RSSTensor<T> &res, LUT_Param<T> &parameter, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    uint32_t size = x.size(), table_size = parameter.table_size;
    // 1. Compute x_hat = x + r
    RSSTensor<T> x_hat(x.shape);
    // Add r (RSS) to x (RSS). r is stored in parameter.r_share
#pragma omp parallel for
    for(uint32_t i=0; i<size; ++i) {
        x_hat.first.data[i] = x.first.data[i] - parameter.r_in_share[party.party_id];
        x_hat.second.data[i] = x.second.data[i] - parameter.r_in_share[(party.party_id+1) % 3];
    }

    // 2. Open x_hat
    Tensor<T> x_hat_open(x.shape);
    restore(x_hat, x_hat_open, malicious);

    // 3. Evaluation (online): compute <u> = <v> >> x_hat and w = sum_j T[j] * <u[j]>
    //    Here parameter.onehot_table[k] holds party k's RSS-like share of full-domain eval vector <v>.
    //    We compute each party's res.first as convolution with its own share,
    //    and res.second as convolution with next party's share, so that (first,second)
    //    corresponds to RSS layout (P0: (s0,s1), P1: (s1,s2), P2: (s2,s0)).
    res.assert_same_shape(x);
    int pid = party.party_id;
#pragma omp parallel for
    for (uint32_t i = 0; i < size; ++i)
    {
        // reduce x_hat to table domain
        uint64_t shift = (uint64_t)(x_hat_open.data[i]) % (uint64_t)table_size;

        T acc_first = 0;
        T acc_second = 0;
        // accumulate over table entries
        for (uint32_t j = 0; j < table_size; ++j)
        {
            uint32_t idx = (uint32_t)((j + shift) % table_size);
            // v-share for this party and next party
            T v_share_curr = parameter.onehot_table[pid].data[idx];
            T v_share_next = parameter.onehot_table[(pid + 1) % 3].data[idx];
            T tbl_val = parameter.table.data[j];

            acc_first += v_share_curr * tbl_val;
            acc_second += v_share_next * tbl_val;
        }

        res.first.data[i] = acc_first;
        res.second.data[i] = acc_second;
    }
}

template <typename T>
void rss_protocols::lut(RSSTensor<T> &x, RSSTensor<T> &res1, RSSTensor<T> &res2, LUT_Param<T> &parameter1, LUT_Param<T> &parameter2, bool malicious)
{
    // TODO: extend to multi-table
    Party3PC &party = Party3PC::getInstance();
    uint32_t size = x.size();
    uint32_t combined_size = 2 * size;
    
    // 1. Compute x_hat_combined = [x - r1, x - r2]
    RSSTensor<T> x_hat_combined({combined_size});
    int pid = party.party_id;
    int next_pid = (pid + 1) % 3;

    for(uint32_t i=0; i<size; ++i) {
        // First half: x - r1 (for parameter1)
        x_hat_combined.first.data[i] = x.first.data[i] - parameter1.r_in_share[pid];
        x_hat_combined.second.data[i] = x.second.data[i] - parameter1.r_in_share[next_pid];
        
        // Second half: x - r2 (for parameter2)
        x_hat_combined.first.data[i + size] = x.first.data[i] - parameter2.r_in_share[pid];
        x_hat_combined.second.data[i + size] = x.second.data[i] - parameter2.r_in_share[next_pid];
    }

    // 2. Open x_hat (One communication round for both)
    Tensor<T> x_hat_open({combined_size});
    restore(x_hat_combined, x_hat_open, malicious);

    // 3. Evaluation (Online)
    res1.assert_same_shape(x);
    res2.assert_same_shape(x);

#pragma omp parallel for
    for (uint32_t i = 0; i < size; ++i)
    {
        // --- Table 1 Evaluation ---
        uint64_t shift1 = (uint64_t)(x_hat_open.data[i]) % (uint64_t)parameter1.table_size;
        T acc1_first = 0;
        T acc1_second = 0;
        
        for (uint32_t j = 0; j < parameter1.table_size; ++j)
        {
            uint32_t idx = (uint32_t)((j + shift1) % parameter1.table_size);
            T v_curr = parameter1.onehot_table[pid].data[idx];
            T v_next = parameter1.onehot_table[next_pid].data[idx];
            T tbl_val = parameter1.table.data[j];
            
            acc1_first += v_curr * tbl_val;
            acc1_second += v_next * tbl_val;
        }
        res1.first.data[i] = acc1_first;
        res1.second.data[i] = acc1_second;

        // --- Table 2 Evaluation ---
        uint64_t shift2 = (uint64_t)(x_hat_open.data[i + size]) % (uint64_t)parameter2.table_size;
        T acc2_first = 0;
        T acc2_second = 0;
        
        for (uint32_t j = 0; j < parameter2.table_size; ++j)
        {
            uint32_t idx = (uint32_t)((j + shift2) % parameter2.table_size);
            T v_curr = parameter2.onehot_table[pid].data[idx];
            T v_next = parameter2.onehot_table[next_pid].data[idx];
            T tbl_val = parameter2.table.data[j];
            
            acc2_first += v_curr * tbl_val;
            acc2_second += v_next * tbl_val;
        }
        res2.first.data[i] = acc2_first;
        res2.second.data[i] = acc2_second;
    }
}

template <typename T>
void rss_protocols::utils::getk(RSSTensor<T> &x, RSSTensor<T> &k, Parameters<T> &parameters, bool malicious)
{
    // $b ^ k \le x < b ^ {k + 1}$ find k and calculate k + 1
    Party3PC &party = Party3PC::getInstance();
    uint32_t size = x.size();
    uint32_t nexpb_size = (int)(log(pow(2, 2 * x.float_scale_bit)) / log(SCALE_BASE));
    RSSTensor<T> delta({size, nexpb_size}); // x - b ^ i for i from 1 to max size
    int party_id = party.party_id;
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < nexpb_size; j++)
        {   
            T val = (T)(pow(SCALE_BASE, j + 1));
            int idx = i * nexpb_size + j;
            
            delta.first.data[idx] = x.first.data[i];
            delta.second.data[idx] = x.second.data[i];
            
            if (party_id == 0)
                delta.first.data[idx] -= val;
            else if (party_id == 2)
                delta.second.data[idx] -= val;
        }
    }
    nonNegative(delta, delta, parameters, false, malicious);
    // calculate k + 1
    // Sum delta along the second dimension and add 1
    int pid = party.party_id;
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        T sum_first = 0;
        T sum_second = 0;
        for (int j = 0; j < nexpb_size; j++)
        {
            sum_first += delta.first.data[i * nexpb_size + j];
            sum_second += delta.second.data[i * nexpb_size + j];
        }

        if (pid == 0)
        {
            sum_first += 1;
        }
        else if (pid == 2)
        {
            sum_second += 1;
        }
        k.first.data[i] = sum_first;
        k.second.data[i] = sum_second;
    }
 }

template <typename T>
void rss_protocols::inv(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
    RSSTensor<T> k(x.shape);
    utils::getk(x, k, parameters, malicious);

    // calculate b^(-(k+1))
    lut(k, k, parameters.nexpb_param, malicious);

    RSSTensor<T> b(x.shape);
    mul(x, k, b, true, malicious); // b  = x * b^(-(k+1))
    sub(b, (T)((1.0 / SCALE_BASE) * (1 << x.float_scale_bit)), b);
    lut(b, b, parameters.inv_param, malicious);
    mul(k, b, res, true, malicious);
}

template <typename T>
void rss_protocols::rsqrt(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
    RSSTensor<T> k(x.shape);
    utils::getk(x, k, parameters, malicious);

    // calculate b^(-(k+1)) and b^(-1/2(k+1))
    RSSTensor<T> sqrt_nexpbk(k.shape), nexpbk(k.shape);
    lut(k, sqrt_nexpbk, nexpbk, parameters.sqrt_nexpb_param, parameters.nexpb_param, malicious);

    RSSTensor<T> b(x.shape);
    mul(x, nexpbk, b, true, malicious); // b  = x * b^(-(k+1))
    sub(b, (T)((1.0 / SCALE_BASE) * (1 << x.float_scale_bit)), b);
    lut(b, b, parameters.rsqrt_param, malicious);
    mul(sqrt_nexpbk, b, res, true, malicious);
}

template <typename T>
void rss_protocols::utils::gelu_same_scale(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
    T table_size = parameters.gelu_param.table_size;
    uint32_t size = x.size();
    RSSTensor<T> y(x.shape);

    RSSTensor<T> dx(x.shape); // dx = relu(x)
    rss_protocols::select(x, x, dx, parameters, malicious);

    RSSTensor<T> abs_x(x.shape), sizeSubAbs(x.shape);
    rss_protocols::mulConstSubBias(dx, (T)2, x, abs_x); // calculate abs_x = dx * 2 - x

    rss_protocols::sub(table_size, abs_x, sizeSubAbs);
    RSSTensor<T> ia(x.shape);
    rss_protocols::select(sizeSubAbs, abs_x, ia, parameters, malicious);
    rss_protocols::add(ia, (T)(table_size - 1), ia);
    RSSTensor<T> c(x.shape);
    rss_protocols::lut(ia, c, parameters.gelu_param, malicious);
    rss_protocols::sub(dx, c, res);
}

template <typename T>
void rss_protocols::gelu(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
    if (x.float_scale_bit == GELU_TABLE_PRECISION)
    {
        utils::gelu_same_scale(x, res, parameters, malicious);
    }
    else
    {
        T table_size = parameters.gelu_param.table_size;
        uint32_t size = x.size();
        RSSTensor<T> y(x.shape);
        truncate(x, y, 1 << (x.float_scale_bit - GELU_TABLE_PRECISION), malicious);

        RSSTensor<T> x_and_y({2, size});
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            x_and_y.first.data[i] = x.first.data[i];
            x_and_y.second.data[i] = x.second.data[i];
            x_and_y.first.data[size + i] = y.first.data[i];
            x_and_y.second.data[size + i] = y.second.data[i];
        }

        RSSTensor<T> dx_and_dy({2, size}); // dx = relu(x), dy = relu(y)
        select(y, x_and_y, dx_and_dy, 2, parameters, malicious);
        RSSTensor<T> dx(x.shape), dy(x.shape);

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            dx.first.data[i] = dx_and_dy.first.data[i];
            dx.second.data[i] = dx_and_dy.second.data[i];
            dy.first.data[i] = dx_and_dy.first.data[size + i];
            dy.second.data[i] = dx_and_dy.second.data[size + i];
        }

        RSSTensor<T> abs_y(x.shape), sizeSubAbs(x.shape);
        mulConstSubBias(dy, (T)2, y, abs_y); // calculate abs_y = dy * 2 - y

        sub(table_size, abs_y, sizeSubAbs);
        RSSTensor<T> ia(x.shape);
        select(sizeSubAbs, abs_y, ia, parameters, malicious);
        add(ia, (T)(table_size - 1), ia);
        RSSTensor<T> c(x.shape);
        lut(ia, c, parameters.gelu_param, malicious);
        sub(dx, c, res);
    }
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

#pragma omp parallel for private(index0, index1)
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
    T ln2 = (T)(int)floor(log(2) * (1 << x.float_scale_bit));

    uint32_t size = x.size();
    RSSTensor<T> z(x.shape), p(x.shape), p2(x.shape), neg_exp2_z(x.shape), exp_p(x.shape);

    mulConst(x, (T)(-1), z);
    truncate(z, ln2, malicious);                                // z = -x / ln2
    mulConstAddBias(z, ln2, x, p);                              // p = z * ln2 + x
    add(p, (T)(int)floor(1.353 * (1 << x.float_scale_bit)), p); // p + 1.353
    square(p, p2, true, malicious);                             // (p + 1.353) ^ 2
    mulConst(p2, (T)(int)floor(0.3585 * (1 << x.float_scale_bit)), exp_p);
    truncate(exp_p, malicious);
    add(exp_p, (T)(int)floor(0.344 * (1 << x.float_scale_bit)), exp_p); // 0.3585 * (p + 1.353) ^ 2 + 0.344

    // clip z by choose minimum one between z and scale
    RSSTensor<T> z_minus_scale(z.shape), scale_minus_z(z.shape), min_s_z(z.shape);
    sub((T)x.float_scale_bit, z, scale_minus_z);
    sub(z, (T)x.float_scale_bit, z_minus_scale);
    select(z_minus_scale, scale_minus_z, min_s_z, parameter, malicious);

    if (Party3PC::getInstance().party_id == 0)
    {
    #pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            z.first.data[i] += min_s_z.first.data[i] + (T)x.float_scale_bit;
            z.second.data[i] += min_s_z.second.data[i];
        }
    }
    else if (Party3PC::getInstance().party_id == 1)
    {
    #pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            z.first.data[i] += min_s_z.first.data[i];
            z.second.data[i] += min_s_z.second.data[i];
        }
    }
    else
    {
    #pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            z.first.data[i] += min_s_z.first.data[i];
            z.second.data[i] += min_s_z.second.data[i] + (T)x.float_scale_bit;
        }
    }

    lut(z, neg_exp2_z, parameter.nexp2_param, malicious); // 2^(-k)
    mul(exp_p, neg_exp2_z, res, true, malicious);
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
#pragma omp parallel for
    for (int i = 0; i < common_size; i++)
    {
        for (int j = 0; j < dim_size; j++)
        {
            index = i * dim_size + j;
            delta.first.data[index] = x.first.data[index] - x_max.first.data[i];
            delta.second.data[index] = x.second.data[index] - x_max.second.data[i];
        }
    }

    RSSTensor<T> exp_x(x.shape);
    neg_exp(delta, exp_x, parameter, malicious);
    RSSTensor<T> sum(sum_shape);

#pragma omp parallel for
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
#pragma omp parallel
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

template <typename T, typename U>
void rss_protocols::downcast(RSSTensor<T> &x, RSSTensor<U> &res)
{
    int bit_len = sizeof(U) * 8;
#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = (x.first.data[i] >> (x.float_scale_bit - res.float_scale_bit)) % ((uint64_t)1 << bit_len);
        res.second.data[i] = (x.second.data[i] >> (x.float_scale_bit - res.float_scale_bit)) % ((uint64_t)1 << bit_len);
    }
}

template <typename U, typename T>
void rss_protocols::upcast(RSSTensor<U> &x, RSSTensor<T> &res, int party_id, bool malicious)
{
    RSSTensor<U> r(x.shape);
    RSSTensor<T> r_upper(x.shape);
    RSSTensor<T> s(x.shape); // s is the most significant bit of r_upper
    r.zeros();
    r_upper.zeros();
    s.zeros();
    uint32_t size = x.size();
    int down_bit_len = sizeof(U) * 8;

    U bias = 1 << (down_bit_len - 2);
    uint64_t down_ring_max = 1ULL << down_bit_len;
    uint64_t scale_delta = 1 << (res.float_scale_bit - x.float_scale_bit);
    uint64_t w_first, w_second;
    uint64_t is_x_hat_non_neg;
    RSSTensor<U> x_hat(x.shape);
    Tensor<U> x_hat_open(x.shape);

    if (party_id == 0)
    {
    #pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            x_hat.first.data[i] = x.first.data[i] + r.first.data[i] + bias;
            x_hat.second.data[i] = x.second.data[i] + r.second.data[i];
        }

        rss_protocols::restore(x_hat, x_hat_open, malicious);
    #pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            is_x_hat_non_neg = (1 - (x_hat_open.data[i] >> (down_bit_len - 1)));
            w_first = s.first.data[i] * is_x_hat_non_neg;
            w_second = s.second.data[i] * is_x_hat_non_neg;

            res.first.data[i] = (x_hat_open.data[i] - r_upper.first.data[i] + w_first * down_ring_max - bias) * scale_delta;
            res.second.data[i] = (-r_upper.first.data[i] + w_second * down_ring_max) * scale_delta;
        }
    }
    else if (party_id == 1)
    {   
    #pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            x_hat.first.data[i] = x.first.data[i] + r.first.data[i];
            x_hat.second.data[i] = x.second.data[i] + r.second.data[i];
        }

        rss_protocols::restore(x_hat, x_hat_open, malicious);
    #pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            is_x_hat_non_neg = (1 - (x_hat_open.data[i] >> (down_bit_len - 1)));
            w_first = s.first.data[i] * is_x_hat_non_neg;
            w_second = s.second.data[i] * is_x_hat_non_neg;

            res.first.data[i] = (-r_upper.first.data[i] + w_first * down_ring_max) * scale_delta;
            res.second.data[i] = (-r_upper.first.data[i] + w_second * down_ring_max) * scale_delta;
        }
    }
    else
    {
    #pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            x_hat.first.data[i] = x.first.data[i] + r.first.data[i];
            x_hat.second.data[i] = x.second.data[i] + r.second.data[i] + bias;
        }

        rss_protocols::restore(x_hat, x_hat_open, malicious);
    #pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            is_x_hat_non_neg = (1 - (x_hat_open.data[i] >> (down_bit_len - 1)));
            w_first = s.first.data[i] * is_x_hat_non_neg;
            w_second = s.second.data[i] * is_x_hat_non_neg;

            res.first.data[i] = (-r_upper.first.data[i] + w_first * down_ring_max) * scale_delta;
            res.second.data[i] = (x_hat_open.data[i] - r_upper.first.data[i] + w_second * down_ring_max - bias) * scale_delta;
        }
    }
}

