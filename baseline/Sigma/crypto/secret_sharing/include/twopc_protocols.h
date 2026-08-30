#pragma once

#include <fstream>
#include <chrono>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <array>
#include <map>
#include <mutex>
#include <omp.h>
#include <stdexcept>
#include "globals.h"
#include "perf_stats.h"
#include "params.h"
#include "party2pc.h"
#include <thread>
#include "ass_protocols.h"

namespace twopc_protocols
{
    namespace profile
    {
        struct TimingSnapshot
        {
            uint64_t softmax_ns;
            uint64_t softmax_calls;
        };

        inline std::atomic<uint64_t> softmax_ns_total{0};
        inline std::atomic<uint64_t> softmax_calls_total{0};

        inline void add_softmax_ns(uint64_t ns)
        {
            softmax_ns_total.fetch_add(ns, std::memory_order_relaxed);
            softmax_calls_total.fetch_add(1, std::memory_order_relaxed);
        }

        inline void reset()
        {
            softmax_ns_total.store(0, std::memory_order_relaxed);
            softmax_calls_total.store(0, std::memory_order_relaxed);
        }

        inline TimingSnapshot snapshot()
        {
            return TimingSnapshot{
                softmax_ns_total.load(std::memory_order_relaxed),
                softmax_calls_total.load(std::memory_order_relaxed)};
        }
    }

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

    template <typename T>
    class ShareTensor
    {
    public:
        // 2PC additive share: each party keeps one local share.
        Tensor<T> share;
        std::vector<uint32_t> shape;

        static constexpr auto float_scale_bit = kFloat_Precision<T>;

        ShareTensor() = default;

        ShareTensor(const ShareTensor &other)
        {
            shape = other.shape;
            share = other.share;
        }

        explicit ShareTensor(const std::vector<uint32_t> &shape)
        {
            allocate(shape);
        }

        void reshape(const std::vector<uint32_t> &new_shape)
        {
            share.reshape(new_shape);
            shape = new_shape;
        }

        void allocate(const std::vector<uint32_t> &new_shape)
        {
            shape = new_shape;
            share.allocate(shape);
        }

        void free()
        {
            shape = {};
            share.free();
        }

        void rand()
        {
            share.rand();
        }

        void zeros()
        {
            share.zero();
        }

        void fill(T x)
        {
            share.fill(x);
        }

        uint32_t size() const
        {
            return share.size();
        }

        bool is_same_shape(const ShareTensor<T> &other) const
        {
            return share.is_same_shape(other.share);
        }

        bool is_same_shape(const Tensor<T> &other) const
        {
            return share.is_same_shape(other);
        }

        void assert_same_shape(const ShareTensor<T> &other)
        {
            share.assert_same_shape(other.share);
        }

        void assert_same_shape(const Tensor<T> &other)
        {
            share.assert_same_shape(other);
        }

        ShareTensor<T> &operator=(const ShareTensor<T> &other)
        {
            if (this != &other)
            {
                shape = other.shape;
                share = other.share;
            }
            return *this;
        }

        ShareTensor<T> transpose()
        {
            always_assert(shape.size() >= 2);
            std::vector<uint32_t> new_shape = shape;
            std::swap(new_shape[new_shape.size() - 1], new_shape[new_shape.size() - 2]);
            ShareTensor<T> res(new_shape);

            uint32_t freeze_size = 1;
            const uint32_t transpose_size = shape[shape.size() - 1] * shape[shape.size() - 2];
            for (uint32_t i = 0; i + 2 < shape.size(); i++)
            {
                freeze_size *= shape[i];
            }

            for (uint32_t i = 0; i < freeze_size; i++)
            {
                for (uint32_t j = 0; j < shape[shape.size() - 1]; j++)
                {
                    for (uint32_t k = 0; k < shape[shape.size() - 2]; k++)
                    {
                        res.share.data[i * transpose_size + j * shape[shape.size() - 2] + k] =
                            share.data[i * transpose_size + k * shape[shape.size() - 1] + j];
                    }
                }
            }
            return res;
        }

        ShareTensor<T> permute(const std::vector<uint32_t> &permute_dims)
        {
            always_assert(permute_dims.size() == shape.size() && permute_dims.size() > 1);
            const uint32_t shape_size = shape.size();
            for (uint32_t i = 0; i < shape_size; i++)
            {
                always_assert(permute_dims[i] < shape_size);
            }

            std::vector<uint32_t> new_shape = shape;
            for (uint32_t i = 0; i < shape_size; i++)
            {
                new_shape[i] = shape[permute_dims[i]];
            }

            std::vector<uint32_t> idx_base(shape_size, 1);
            for (int i = static_cast<int>(shape_size) - 2; i >= 0; i--)
            {
                idx_base[static_cast<uint32_t>(i)] = idx_base[static_cast<uint32_t>(i + 1)] * shape[static_cast<uint32_t>(i + 1)];
            }

            std::vector<uint32_t> new_idx_base = idx_base;
            for (uint32_t i = 0; i < shape_size; i++)
            {
                new_idx_base[i] = idx_base[permute_dims[i]];
            }

            ShareTensor<T> res(new_shape);
            for (uint32_t i = 0; i < size(); i++)
            {
                uint32_t idx = 0;
                uint32_t temp = i;
                for (int j = static_cast<int>(shape_size) - 1; j >= 0; j--)
                {
                    idx += (temp % new_shape[static_cast<uint32_t>(j)]) * new_idx_base[static_cast<uint32_t>(j)];
                    temp /= new_shape[static_cast<uint32_t>(j)];
                }
                res.share.data[i] = share.data[idx];
            }
            return res;
        }
    };

    // Split a clear tensor into local additive shares and distribute the peer share.
    template <typename T>
    void share(Tensor<T> &x, ShareTensor<T> &res);

    // Receive a previously distributed additive share from another party.
    template <typename T>
    void recv_shares_from(int source_id, ShareTensor<T> &res);

    namespace detail
    {
        // Read an environment variable into a string and report whether it is available.
        inline bool try_get_env(const char *env_name, std::string &value)
        {
            const char *env = std::getenv(env_name);
            if (env == nullptr || env[0] == '\0')
            {
                return false;
            }
            value = std::string(env);
            return true;
        }

        // Deserialize one tensor worth of integer values from a text stream.
        template <typename T>
        inline bool read_tensor_from_stream(std::ifstream &in, Tensor<T> &out)
        {
            int64_t value = 0;
            for (uint32_t i = 0; i < out.size(); i++)
            {
                if (!(in >> value))
                {
                    return false;
                }
                out.data[i] = static_cast<T>(value);
            }
            return true;
        }

        // Load one Beaver triple from a text file, or fall back to an all-zero triple if the env var is unset.
        template <typename T>
        inline void load_offline_triple_from_file(const char *env_name, Tensor<T> &a, Tensor<T> &b, Tensor<T> &c)
        {
            static std::mutex io_mutex;
            static std::map<std::string, uint64_t> consumed_values;

            std::lock_guard<std::mutex> guard(io_mutex);
            std::string file_path;
            if (!try_get_env(env_name, file_path))
            {
                // Default to a zero triple when offline files are not configured.
                a.zero();
                b.zero();
                c.zero();
                return;
            }

            std::ifstream in(file_path);
            if (!in.is_open())
            {
                throw std::runtime_error("Failed to open offline triple file: " + file_path);
            }

            uint64_t skip = consumed_values[file_path];
            int64_t dummy = 0;
            while (skip > 0)
            {
                if (!(in >> dummy))
                {
                    throw std::runtime_error("Offline triple file exhausted while skipping consumed values: " + file_path);
                }
                skip--;
            }

            if (!read_tensor_from_stream(in, a) || !read_tensor_from_stream(in, b) || !read_tensor_from_stream(in, c))
            {
                throw std::runtime_error("Offline triple file does not contain enough values for requested triple: " + file_path);
            }

            consumed_values[file_path] += static_cast<uint64_t>(a.size()) + static_cast<uint64_t>(b.size()) + static_cast<uint64_t>(c.size());
        }

        // Interpret a ring element as its signed integer representative.
        template <typename T>
        inline int64_t ring_to_signed(const T v)
        {
            if constexpr (sizeof(T) == 8)
            {
                return static_cast<int64_t>(v);
            }
            else
            {
                constexpr int bits = static_cast<int>(sizeof(T) * 8);
                const uint64_t uv = static_cast<uint64_t>(v);
                const uint64_t sign = (1ULL << (bits - 1));
                if (uv & sign)
                {
                    return static_cast<int64_t>(uv) - static_cast<int64_t>(1ULL << bits);
                }
                return static_cast<int64_t>(uv);
            }
        }

        // Map a signed integer back into the ring domain used by T.
        template <typename T>
        inline T signed_to_ring(const int64_t v)
        {
            if constexpr (sizeof(T) == 8)
            {
                return static_cast<T>(v);
            }
            else
            {
                constexpr int bits = static_cast<int>(sizeof(T) * 8);
                const int64_t mod = static_cast<int64_t>(1ULL << bits);
                int64_t r = v % mod;
                if (r < 0)
                {
                    r += mod;
                }
                return static_cast<T>(r);
            }
        }

        // Quantize a floating-point value into the integer ring.
        template <typename T>
        inline T double_to_ring(const double v)
        {
            return signed_to_ring<T>(static_cast<int64_t>(std::llround(v)));
        }

        // Decode a fixed-point ring value into a double.
        template <typename T>
        inline double ring_to_double_fixed(const T v)
        {
            const double scale = static_cast<double>(1ULL << kFloat_Precision<T>);
            return static_cast<double>(ring_to_signed(v)) / scale;
        }

        // Encode a double into the fixed-point representation used by T.
        template <typename T>
        inline T fixed_to_ring(const double v)
        {
            const double scale = static_cast<double>(1ULL << kFloat_Precision<T>);
            return double_to_ring<T>(v * scale);
        }

        // Test the sign bit directly in ring representation.
        template <typename T>
        inline bool is_non_negative_ring(const T v)
        {
            constexpr int bits = static_cast<int>(sizeof(T) * 8);
            return ((static_cast<uint64_t>(v) >> (bits - 1)) == 0);
        }


        // Publish a clear tensor back into the shared domain by resharing from party 0.
        template <typename T>
        void publish_clear_result_to_share(const Tensor<T> &clear_res, ShareTensor<T> &share_res)
        {
            Party2PC &party = Party2PC::getInstance();
            if (party.party_id == 0)
            {
                Tensor<T> tmp(clear_res.shape);
                tmp.copy(clear_res);
                twopc_protocols::share(tmp, share_res);
            }
            else
            {
                twopc_protocols::recv_shares_from(0, share_res);
            }
        }
    }

    namespace debug
    {
        // Open and print the full shared tensor as real values.
        template <typename T>
        void openPrintReal(ShareTensor<T> &x);

        // Open and print a single element from a shared tensor.
        template <typename T>
        void openPrintReal(ShareTensor<T> &x, int index);

        // Open and print a selected set of tensor indices.
        template <typename T>
        void openPrintReal(ShareTensor<T> &x, std::vector<int> index);

        // Open and print a half-open interval of tensor elements.
        template <typename T>
        void openPrintReal(ShareTensor<T> &x, int start, int end);

        // Materialize a shared tensor and dump it to a file.
        template <typename T>
        void printRealToFile(ShareTensor<T> &x, const std::string &file_name);
    }

    namespace utils
    {
        // Shared-shared batched matrix multiplication into a clear tensor buffer.
        template <typename T>
        void SSMatMul(ShareTensor<T> &x, ShareTensor<T> &y, Tensor<T> &res, const uint32_t broadcast_size,
                       const uint32_t common_size, const uint32_t row, const uint32_t common_dim, const uint32_t col,
                       const bool is_b_broadcast);

        // Shared-clear batched matrix multiplication into a shared tensor buffer.
        template <typename T>
        void SSMatMul(ShareTensor<T> &x, Tensor<T> &y, ShareTensor<T> &res, const uint32_t broadcast_size,
                       const uint32_t common_size, const uint32_t row, const uint32_t common_dim, const uint32_t col,
                       const bool is_b_broadcast);

        // Clear-shared batched matrix multiplication into a shared tensor buffer.
        template <typename T>
        void SSMatMul(Tensor<T> &x, ShareTensor<T> &y, ShareTensor<T> &res, const uint32_t broadcast_size,
                       const uint32_t common_size, const uint32_t row, const uint32_t common_dim, const uint32_t col,
                       const bool is_b_broadcast);

        // Private comparison primitive used by sign and comparison protocols.
        template <typename T>
        void privateCompare(Tensor<T> &x_with_pre, Tensor<T> &x_with_next, Tensor<T> &res_with_pre,
                            Tensor<T> &res_with_next, Parameters<T> &parameter);

        // Extract the most significant comparison bit as a shared tensor.
        template <typename T>
        void getk(ShareTensor<T> &x, ShareTensor<T> &k, Parameters<T> &parameter, bool malicious = false, uint32_t eval_bitlength = 0);

        // GELU helper that preserves the original scaling factor.
        template <typename T>
        void gelu_same_scale(ShareTensor<T> &x, ShareTensor<T> &res, Parameters<T> &parameters, bool malicious = false);
    }

    // Reconstruct a shared tensor into a clear tensor on the calling party.
    template <typename T>
    void restore(ShareTensor<T> &x, Tensor<T> &res, bool malicious = false);

    // Reconstruct a shared tensor only on the designated target party.
    template <typename T>
    void reconstruct_to(int target_id, ShareTensor<T> &x, Tensor<T> &res = NULL, bool malicious = false);

    // Split a clear tensor into additive shares.
    template <typename T>
    void share(Tensor<T> &x, ShareTensor<T> &res);

    // Receive an additive share tensor from another party.
    template <typename T>
    void recv_shares_from(int source_id, ShareTensor<T> &res);

    // Produce a shared random coin tensor.
    template <typename T>
    void coin(std::vector<uint32_t> shape, ShareTensor<T> &res);

    // Convert a clear tensor back into a shared tensor.
    template <typename T>
    void reshare(Tensor<T> &x, ShareTensor<T> &res);

    // Return-by-value version of reshare.
    template <typename T>
    ShareTensor<T> reshare(Tensor<T> &x);

    // Shared plus shared addition.
    template <typename T>
    void add(ShareTensor<T> &x, ShareTensor<T> &y, ShareTensor<T> &res);

    // Shared plus clear tensor addition.
    template <typename T>
    void add(ShareTensor<T> &x, Tensor<T> &y, ShareTensor<T> &res);

    // Shared plus scalar addition.
    template <typename T>
    void add(ShareTensor<T> &x, T y, ShareTensor<T> &res);

    // Shared minus shared subtraction.
    template <typename T>
    void sub(ShareTensor<T> &x, ShareTensor<T> &y, ShareTensor<T> &res);

    // Shared minus clear tensor subtraction.
    template <typename T>
    void sub(ShareTensor<T> &x, Tensor<T> &y, ShareTensor<T> &res);

    // Shared minus scalar subtraction.
    template <typename T>
    void sub(ShareTensor<T> &x, T y, ShareTensor<T> &res);

    // Scalar minus shared subtraction.
    template <typename T>
    void sub(T x, ShareTensor<T> &y, ShareTensor<T> &res);

    // Multiply a shared tensor by a clear tensor element-wise.
    template <typename T>
    void mulConst(ShareTensor<T> &x, Tensor<T> &y, ShareTensor<T> &res);

    // Multiply a shared tensor by a scalar.
    template <typename T>
    void mulConst(ShareTensor<T> &x, T y, ShareTensor<T> &res);

    // Multiply by a scalar and add a public scalar bias.
    template <typename T>
    void mulConstAddBias(ShareTensor<T> &x, T y, T bias, ShareTensor<T> &res);

    // Multiply by a scalar and add a shared bias tensor.
    template <typename T>
    void mulConstAddBias(ShareTensor<T> &x, T y, ShareTensor<T> &bias, ShareTensor<T> &res);

    // Multiply by a clear tensor and add a shared bias tensor.
    template <typename T>
    void mulConstAddBias(ShareTensor<T> &x, Tensor<T> &y, ShareTensor<T> &bias, ShareTensor<T> &res);

    // Multiply by a scalar and subtract a public scalar bias.
    template <typename T>
    void mulConstSubBias(ShareTensor<T> &x, T y, T bias, ShareTensor<T> &res);

    // Multiply by a scalar and subtract a shared bias tensor.
    template <typename T>
    void mulConstSubBias(ShareTensor<T> &x, T y, ShareTensor<T> &bias, ShareTensor<T> &res);

    // Multiply by a clear tensor and subtract a shared bias tensor.
    template <typename T>
    void mulConstSubBias(ShareTensor<T> &x, Tensor<T> &y, ShareTensor<T> &bias, ShareTensor<T> &res);

    // Secure element-wise multiplication using Beaver triples.
    template <typename T>
    void mul(ShareTensor<T> &x, ShareTensor<T> &y, ShareTensor<T> &res, bool needTrunc = false, bool malicious = false);

    // Multiply a shared tensor by a two-part constant packed in a pair.
    template <typename T>
    void mul(ShareTensor<T> &x, std::pair<T, T> &y, ShareTensor<T> &res);

    // Secure squaring implemented through secure multiplication.
    template <typename T>
    void square(ShareTensor<T> &x, ShareTensor<T> &res, bool needTrunc = false, bool malicious = false);

    // Secure matrix multiplication using Beaver triples.
    template <typename T>
    void matMul(ShareTensor<T> &x, ShareTensor<T> &y, ShareTensor<T> &res, bool needTrunc = false, bool malicious = false);

    template <typename T>
    void truncate(ShareTensor<T> &x, ShareTensor<T> &res, size_t scale, bool malicious = false);

    template <typename T>
    void truncate(ShareTensor<T> &x, size_t scale, bool malicious = false);

    template <typename T>
    void truncate(ShareTensor<T> &x, bool malicious = false);

    template <typename T>
    void checkZero(ShareTensor<T> &x);

    template <typename T>
    void macCheck(const ShareTensor<T> &x, const ShareTensor<T> &mx, const std::pair<T, T> &mac_key);

    template <typename T>
    void macCheck(ShareTensor<T> &x, ShareTensor<T> &mx, ShareTensor<T> &mac_key);

    template <typename T>
    void macCheckSimulate(uint32_t size);

    template <typename T>
    void pc_msb(ShareTensor<T> &x, ShareTensor<T> &res,
                Parameters<T> &parameter, const uint32_t size, bool malicious = false, const char *perf_scope_prefix = nullptr, uint32_t eval_bitlength = 0);

    template <typename T>
    void nonNegative(ShareTensor<T> &x, ShareTensor<T> &res, Parameters<T> &parameter, bool isFloat = true, bool malicious = false, const char *perf_scope_prefix = nullptr, uint32_t eval_bitlength = 0);

    template <typename T>
    void greaterEqual(ShareTensor<T> &x, ShareTensor<T> &y, ShareTensor<T> &res, Parameters<T> &parameter, bool isFloat = true, bool malicious = false);

    template <typename T>
    void select(ShareTensor<T> &x, ShareTensor<T> &y, ShareTensor<T> &res, Parameters<T> &parameter, bool malicious = false, const char *perf_scope_prefix = nullptr, uint32_t eval_bitlength = 0);

    template <typename T>
    void select(ShareTensor<T> &x, ShareTensor<T> &y, ShareTensor<T> &res, uint32_t y_num, Parameters<T> &parameter, bool malicious = false, const char *perf_scope_prefix = nullptr, uint32_t eval_bitlength = 0);

    template <typename T>
    void lut(ShareTensor<T> &x, ShareTensor<T> &res, LUT_Param<T> &parameter, bool malicious = false);

    template <typename T>
    void lut(ShareTensor<T> &x, ShareTensor<T> &res1, ShareTensor<T> &res2, LUT_Param<T> &parameter1, LUT_Param<T> &parameter2, bool malicious = false);

    template <typename T>
    void inv(ShareTensor<T> &x, ShareTensor<T> &res, uint32_t dim_size, Parameters<T> &parameters, bool malicious = false);

    template <typename T>
    void rsqrt(ShareTensor<T> &x, ShareTensor<T> &res, Parameters<T> &parameters, bool malicious = false);

    template <typename T>
    void gelu(ShareTensor<T> &x, ShareTensor<T> &res, Parameters<T> &parameters, bool malicious = false);

    template <typename T>
    void silu(ShareTensor<T> &x, ShareTensor<T> &res, Parameters<T> &parameters, bool malicious = false);

    template <typename T>
    void swiglu(ShareTensor<T> &gate, ShareTensor<T> &up, ShareTensor<T> &res, Parameters<T> &parameters, bool malicious = false);

    template <typename T>
    void rms_norm(ShareTensor<T> &x, ShareTensor<T> &weight, ShareTensor<T> &res, Parameters<T> &parameters, bool malicious = false);

    template <typename T>
    void max_last_dim(ShareTensor<T> &x, ShareTensor<T> &res, Parameters<T> &parameters, bool malicious = false);

    template <typename T>
    void neg_exp(ShareTensor<T> &x, ShareTensor<T> &res, Parameters<T> &Parameters, bool malicious = false);

    template <typename T>
    void softmax_forward(ShareTensor<T> &x, ShareTensor<T> &res, Parameters<T> &parameter, bool malicious = false);

    template <typename T>
    void grouped_multi_query_attention(ShareTensor<T> &query, ShareTensor<T> &key, ShareTensor<T> &value, ShareTensor<T> &res,
                                       Parameters<T> &parameter, uint32_t num_heads,
                                       uint32_t num_key_value_heads, bool masked = true, bool malicious = false);
}

template <typename T>
void twopc_protocols::debug::openPrintReal(ShareTensor<T> &x)
{
    openPrintReal(x, 0, x.size());
}

template <typename T>
void twopc_protocols::debug::openPrintReal(ShareTensor<T> &x, int index)
{
    openPrintReal(x, index, index + 1);
}

template <typename T>
void twopc_protocols::debug::openPrintReal(ShareTensor<T> &x, std::vector<int> index)
{
    Tensor<T> real_res(x.shape);
    restore(x, real_res);

    if (Party2PC::getInstance().party_id == 0)
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
void twopc_protocols::debug::openPrintReal(ShareTensor<T> &x, int start, int end)
{
    always_assert(start < end);
    Tensor<T> real_res(x.shape);
    restore(x, real_res);

    if (Party2PC::getInstance().party_id == 0)
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
void twopc_protocols::debug::printRealToFile(ShareTensor<T> &x, const std::string &file_name)
{
    Tensor<T> real_res(x.shape);
    restore(x, real_res);

    if (Party2PC::getInstance().party_id == 0)
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

// 共享张量乘共享张量的底层本地矩阵乘内核，直接按索引做 batched matmul 并写入明文缓冲区。
template <typename T>
void twopc_protocols::utils::SSMatMul(ShareTensor<T> &x, ShareTensor<T> &y, Tensor<T> &res, const uint32_t broadcast_size,
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

                        T sum = 0;
                        for (uint32_t m = 0; m < common_dim; m++)
                        {
                            sum += x.share.data[index_a + m] * y.share.data[index_b + m * col];
                        }
                        res.data[index] = sum;
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

                        T sum = 0;
                        for (uint32_t m = 0; m < common_dim; m++)
                        {
                            sum += x.share.data[index_a + m] * y.share.data[index_b + m * col];
                        }
                        res.data[index] = sum;
                    }
                }
            }
        }
    }
}

// 共享张量乘明文张量的底层本地矩阵乘内核，结果仍保存在共享张量的本地 share 中。
template <typename T>
void twopc_protocols::utils::SSMatMul(ShareTensor<T> &x, Tensor<T> &y, ShareTensor<T> &res, const uint32_t broadcast_size,
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

                        T sum = 0;
                        for (uint32_t m = 0; m < common_dim; m++)
                        {
                            sum += x.share.data[index_a + m] * y.data[index_b + m * col];
                        }
                        res.share.data[index] = sum;
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

                        T sum = 0;
                        for (uint32_t m = 0; m < common_dim; m++)
                        {
                            sum += x.share.data[index_a + m] * y.data[index_b + m * col];
                        }
                        res.share.data[index] = sum;
                    }
                }
            }
        }
    }
}

// 明文张量乘共享张量的底层本地矩阵乘内核，结果写回共享张量的本地 share。
template <typename T>
void twopc_protocols::utils::SSMatMul(Tensor<T> &x, ShareTensor<T> &y, ShareTensor<T> &res, const uint32_t broadcast_size,
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

                        T sum = 0;
                        for (uint32_t m = 0; m < common_dim; m++)
                        {
                            sum += x.data[index_a + m] * y.share.data[index_b + m * col];
                        }
                        res.share.data[index] = sum;
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

                        T sum = 0;
                        for (uint32_t m = 0; m < common_dim; m++)
                        {
                            sum += x.data[index_a + m] * y.share.data[index_b + m * col];
                        }
                        res.share.data[index] = sum;
                    }
                }
            }
        }
    }
}

// 打开共享张量：把对方 share 收上来，与本地 share 相加恢复出明文结果。
template <typename T>
void twopc_protocols::restore(ShareTensor<T> &x, Tensor<T> &res, bool malicious)
{
    Party2PC &party = Party2PC::getInstance();
    x.assert_same_shape(res);
    // Avoid both parties blocking in send() on large tensors.
    if (party.party_id == 0)
    {
        party.send_tensor_to(party.next_party_id, x.share);
        party.recv_tensor_from(party.next_party_id, res);
    }
    else
    {
        party.recv_tensor_from(party.next_party_id, res);
        party.send_tensor_to(party.next_party_id, x.share);
    }

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.data[i] = res.data[i] + x.share.data[i];
    }
}

// 只向指定参与方重构共享张量，非目标方只负责发送自己的 share。
template <typename T>
void twopc_protocols::reconstruct_to(int target_id, ShareTensor<T> &x, Tensor<T> &res, bool malicious)
{
    Party2PC &party = Party2PC::getInstance();
    if (target_id == party.next_party_id)
    {
        party.send_tensor_to(target_id, x.share);
    }
    else if (target_id == party.party_id)
    {
        party.recv_tensor_from(party.next_party_id, res);

#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.data[i] = res.data[i] + x.share.data[i];
        }
    }
    else
    {
        throw std::invalid_argument("Invalid target_id for 2PC reconstruct_to");
    }
}

// 把明文切成两份加法 share：本地随机采样一份，再把互补份额发给对方。
template <typename T>
void twopc_protocols::share(Tensor<T> &x, ShareTensor<T> &res)
{
    Party2PC &party = Party2PC::getInstance();
    res.assert_same_shape(x);
    res.rand();
    Tensor<T> tmp(x.shape);
#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        tmp.data[i] = x.data[i] - res.share.data[i];
    }

    party.send_tensor_to(party.next_party_id, tmp);
}

// 从指定参与方接收共享张量的本地份额。
template <typename T>
void twopc_protocols::recv_shares_from(int source_id, ShareTensor<T> &res)
{
    Party2PC &party = Party2PC::getInstance();
    party.recv_tensor_from(source_id, res.share);
}

// 共享随机币接口占位，当前尚未实现具体逻辑。
template <typename T>
void twopc_protocols::coin(std::vector<uint32_t> shape, ShareTensor<T> &res)
{
}

// 直接把明文张量拷到本地 share 中，不做通信。
template <typename T>
void twopc_protocols::reshare(Tensor<T> &x, ShareTensor<T> &res)
{
    res.assert_same_shape(x);
    res.share.copy(x);
}

// reshare 的按值返回版本，便于临时构造共享张量。
template <typename T>
twopc_protocols::ShareTensor<T> twopc_protocols::reshare(Tensor<T> &x)
{
    ShareTensor<T> res(x.shape);
    res.share.copy(x);
    return res;
}


// 共享张量加明文张量：只由一个参与方把明文注入到自己的 share 中。
template <typename T>
void twopc_protocols::add(ShareTensor<T> &x, Tensor<T> &y, ShareTensor<T> &res)
{
    int party_id = Party2PC::getInstance().party_id;
    res.assert_same_shape(x);
    res.assert_same_shape(y);

    if (party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.share.data[i] = x.share.data[i] + y.data[i];
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.share.data[i] = x.share.data[i];
            res.share.data[i] = x.share.data[i];
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.share.data[i] = x.share.data[i];
            res.share.data[i] = x.share.data[i] + y.data[i];
        }
    }
}

template <typename T>
void twopc_protocols::add(ShareTensor<T> &x, ShareTensor<T> &y, ShareTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(y);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.share.data[i] = x.share.data[i] + y.share.data[i];
    }
}

// 共享张量加明文标量：只由一个参与方把标量注入到自己的 share 中。
template <typename T>
void twopc_protocols::add(ShareTensor<T> &x, T y, ShareTensor<T> &res)
{
    int party_id = Party2PC::getInstance().party_id;
    res.assert_same_shape(x);

    if (party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.share.data[i] = x.share.data[i] + y;
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.share.data[i] = x.share.data[i];
            res.share.data[i] = x.share.data[i];
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.share.data[i] = x.share.data[i];
            res.share.data[i] = x.share.data[i] + y;
        }
    }
}

// 两个共享张量逐元素相减，本地 share 直接相减即可。
template <typename T>
void twopc_protocols::sub(ShareTensor<T> &x, ShareTensor<T> &y, ShareTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(y);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.share.data[i] = x.share.data[i] - y.share.data[i];
    }
}

// 共享张量减明文张量：只由一个参与方把明文减到自己的 share 中。
template <typename T>
void twopc_protocols::sub(ShareTensor<T> &x, Tensor<T> &y, ShareTensor<T> &res)
{
    int party_id = Party2PC::getInstance().party_id;
    res.assert_same_shape(x);
    res.assert_same_shape(y);

    if (party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.share.data[i] = x.share.data[i] - y.data[i];
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.share.data[i] = x.share.data[i];
            res.share.data[i] = x.share.data[i];
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.share.data[i] = x.share.data[i];
            res.share.data[i] = x.share.data[i] - y.data[i];
        }
    }
}

// 共享张量减明文标量：只由一个参与方把标量减到自己的 share 中。
template <typename T>
void twopc_protocols::sub(ShareTensor<T> &x, T y, ShareTensor<T> &res)
{
    int party_id = Party2PC::getInstance().party_id;
    res.assert_same_shape(x);

    if (party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.share.data[i] = x.share.data[i] - y;
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.share.data[i] = x.share.data[i];
            res.share.data[i] = x.share.data[i];
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.share.data[i] = x.share.data[i];
            res.share.data[i] = x.share.data[i] - y;
        }
    }
}

// 计算“明文标量减共享张量”，通过调整各方本地 share 实现。
template <typename T>
void twopc_protocols::sub(T x, ShareTensor<T> &y, ShareTensor<T> &res)
{
    int party_id = Party2PC::getInstance().party_id;
    res.assert_same_shape(y);

    if (party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < y.size(); i++)
        {
            res.share.data[i] = -y.share.data[i];
            res.share.data[i] += x;
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < y.size(); i++)
        {
            res.share.data[i] = -y.share.data[i];
            res.share.data[i] = -y.share.data[i];
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < y.size(); i++)
        {
            res.share.data[i] = -y.share.data[i];
            res.share.data[i] = x - y.share.data[i];
        }
    }
}

// 共享张量与明文张量逐元素相乘，本地 share 直接乘明文即可。
template <typename T>
void twopc_protocols::mulConst(ShareTensor<T> &x, Tensor<T> &y, ShareTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(y);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.share.data[i] = x.share.data[i] * y.data[i];
    }
}

// 共享张量逐元素乘明文标量，本地 share 直接缩放即可。
template <typename T>
void twopc_protocols::mulConst(ShareTensor<T> &x, T y, ShareTensor<T> &res)
{
    res.assert_same_shape(x);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.share.data[i] = x.share.data[i] * y;
    }
}

// 对共享张量做仿射变换：先乘明文标量，再加明文偏置。
template <typename T>
void twopc_protocols::mulConstAddBias(ShareTensor<T> &x, T y, T bias, ShareTensor<T> &res)
{
    int party_id = Party2PC::getInstance().party_id;
    res.assert_same_shape(x);

    if (party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.share.data[i] = x.share.data[i] * y + bias;
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.share.data[i] = x.share.data[i] * y;
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.share.data[i] = x.share.data[i] * y + bias;
        }
    }
}

// 对共享张量做仿射变换：先乘明文标量，再加共享偏置。
template <typename T>
void twopc_protocols::mulConstAddBias(ShareTensor<T> &x, T y, ShareTensor<T> &bias, ShareTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(bias);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.share.data[i] = x.share.data[i] * y + bias.share.data[i];
    }
}

// 对共享张量做仿射变换：逐元素乘明文张量，再加共享偏置。
template <typename T>
void twopc_protocols::mulConstAddBias(ShareTensor<T> &x, Tensor<T> &y, ShareTensor<T> &bias, ShareTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(y);
    res.assert_same_shape(bias);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.share.data[i] = x.share.data[i] * y.data[i] + bias.share.data[i];
    }
}

// 对共享张量做仿射变换：先乘明文标量，再减明文偏置。
template <typename T>
void twopc_protocols::mulConstSubBias(ShareTensor<T> &x, T y, T bias, ShareTensor<T> &res)
{
    int party_id = Party2PC::getInstance().party_id;
    res.assert_same_shape(x);

    if (party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.share.data[i] = x.share.data[i] * y - bias;
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.share.data[i] = x.share.data[i] * y;
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.share.data[i] = x.share.data[i] * y - bias;
        }
    }
}

// 对共享张量做仿射变换：先乘明文标量，再减共享偏置。
template <typename T>
void twopc_protocols::mulConstSubBias(ShareTensor<T> &x, T y, ShareTensor<T> &bias, ShareTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(bias);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.share.data[i] = x.share.data[i] * y - bias.share.data[i];
    }
}

// 对共享张量做仿射变换：逐元素乘明文张量，再减共享偏置。
template <typename T>
void twopc_protocols::mulConstSubBias(ShareTensor<T> &x, Tensor<T> &y, ShareTensor<T> &bias, ShareTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(y);
    res.assert_same_shape(bias);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.share.data[i] = x.share.data[i] * y.data[i] - bias.share.data[i];
    }
}

// 共享逐元素乘法：按两方 Beaver 公式分别计算本地 share，并把公开的 ef 只注入到一方。
template <typename T>
void twopc_protocols::mul(ShareTensor<T> &x, ShareTensor<T> &y, ShareTensor<T> &res, bool needTrunc, bool malicious)
{
    Party2PC &party = Party2PC::getInstance();
    res.assert_same_shape(x);
    res.assert_same_shape(y);

    ShareTensor<T> a(x.shape), b(y.shape), c(res.shape);
    a.zeros();
    b.zeros();
    c.zeros();

    ShareTensor<T> e_share(x.shape), f_share(y.shape);
    Tensor<T> e(x.shape), f(y.shape), ef(x.shape);
    sub(x, a, e_share);
    sub(y, b, f_share);
    restore(e_share, e, malicious);
    restore(f_share, f, malicious);

    Tensor<T>::mul(ef, e, f);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.share.data[i] = c.share.data[i] + a.share.data[i] * f.data[i] + b.share.data[i] * e.data[i];
        if (party.party_id == 1)
        {
            res.share.data[i] += ef.data[i];
        }
    }

    if (needTrunc)
    {
        truncate(res, malicious);
    }
}

// 把 pair 形式编码的明文常量先求和，再乘到共享张量上。
template <typename T>
void twopc_protocols::mul(ShareTensor<T> &x, std::pair<T, T> &y, ShareTensor<T> &res)
{
    res.assert_same_shape(x);
    uint32_t size = x.size();

    Tensor<T> tmp(x.shape);
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        tmp.data[i] = x.share.data[i] * static_cast<T>(y.first + y.second);
    }

    reshare(tmp, res);
}

// 复用共享乘法来实现平方。
template <typename T>
void twopc_protocols::square(ShareTensor<T> &x, ShareTensor<T> &res, bool needTrunc, bool malicious)
{
    mul(x, x, res, needTrunc, malicious);
}

// 共享矩阵乘法：按两方 Beaver 公式计算 c + a*f + e*b，并把公开的 e*f 只加到一方。
template <typename T>
void twopc_protocols::matMul(ShareTensor<T> &x, ShareTensor<T> &y, ShareTensor<T> &res, bool needTrunc, bool malicious)
{
    perf_stats::Scope perf_scope("twopc_protocols::matMul");
    Party2PC &party = Party2PC::getInstance();

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

    ShareTensor<T> a(x.shape), b(y.shape), c(res.shape);
    a.zeros();
    b.zeros();
    c.zeros();

    ShareTensor<T> e_share(x.shape), f_share(y.shape);
    Tensor<T> e(x.shape), f(y.shape), ef(res.shape);
    sub(x, a, e_share);
    sub(y, b, f_share);
    restore(e_share, e, malicious);
    restore(f_share, f, malicious);

    ShareTensor<T> af(res.shape), eb(res.shape);
    utils::SSMatMul(a, f, af, broadcast_size, common_size, row, common_dim, col, is_b_broadcast);
    utils::SSMatMul(e, b, eb, broadcast_size, common_size, row, common_dim, col, is_b_broadcast);
    Tensor<T>::matmul(ef, e, f, broadcast_size, common_size, row, common_dim, col, is_b_broadcast);

#pragma omp parallel for
    for (int i = 0; i < res.size(); i++)
    {
        res.share.data[i] = c.share.data[i] + af.share.data[i] + eb.share.data[i];
        if (party.party_id == 1)
        {
            res.share.data[i] += ef.data[i];
        }
    }

    if (needTrunc)
    {
        truncate(res, malicious);
    }
}

template <typename T>
void twopc_protocols::truncate(ShareTensor<T> &x, ShareTensor<T> &res, size_t scale, bool malicious)
{
    Party2PC &party = Party2PC::getInstance();
    uint32_t size = x.size();
    Tensor<T> x_shift(x.shape);
    restore(x, x_shift, malicious);

    if (party.party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            if constexpr (std::is_same_v<T, uint32_t>)
            {
                res.share.data[i] = static_cast<T>(static_cast<double>(static_cast<int32_t>(x_shift.data[i])) /
                                                   static_cast<double>(static_cast<int32_t>(scale)));
            }
            else
            {
                res.share.data[i] = static_cast<T>(static_cast<double>(static_cast<int64_t>(x_shift.data[i])) /
                                                   static_cast<double>(static_cast<int64_t>(scale)));
            }
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            res.share.data[i] = 0;
        }
    }
}

template <typename T>
void twopc_protocols::truncate(ShareTensor<T> &x, size_t scale, bool malicious)
{
    truncate(x, x, scale, malicious);
}

template <typename T>
void twopc_protocols::truncate(ShareTensor<T> &x, bool malicious)
{
    truncate(x, 1 << x.float_scale_bit, malicious);
}

template <typename T>
void twopc_protocols::checkZero(ShareTensor<T> &x)
{
    ShareTensor<T> r(x.shape), xr(x.shape);
    Tensor<T> xr_open(x.shape);

    r.zeros(); // it should be random
    mul(x, r, xr);
    restore(xr, xr_open, true);

    xr_open.assert_zero();
}

template <typename T>
void twopc_protocols::macCheck(const ShareTensor<T> &x, const ShareTensor<T> &mx, const std::pair<T, T> &mac_key)
{
#if (IS_MALICIOUS)
    Party2PC &party = Party2PC::getInstance();
    ShareTensor<T> r({1}), mr({1});
    r.zeros(); // it should be random
    mul(r, mac_key, mr);
    ShareTensor<T> ro_share(x.shape);
    ro_share.zeros(); // it should be random
    Tensor<T> ro(x.shape);
    restore(ro_share, ro, true);
    ShareTensor<T> v({1}), w({1});
    v.share.data[0] = r.share.data[0];
    v.share.data[0] = r.share.data[0];

    w.share.data[0] = mr.share.data[0];
    w.share.data[0] = mr.share.data[0];

    for (int i = 0; i < x.size(); i++)
    {
        v.share.data[i] += x.share.data[i] * ro.data[i];
        v.share.data[i] += x.share.data[i] * ro.data[i];
        w.share.data[i] += mx.share.data[i] * ro.data[i];
        w.share.data[i] += mx.share.data[i] * ro.data[i];
    }
    Tensor<T> v_real({1});
    restore(v, v_real, true);
    ShareTensor<T> delta({1});
    mulConstSubBias(mac_key, v_real, w, delta);
    checkZero(delta);
#endif
}

template <typename T>
void twopc_protocols::macCheck(ShareTensor<T> &x, ShareTensor<T> &mx, ShareTensor<T> &mac_key)
{
#if (IS_MALICIOUS)
    ShareTensor<T> r(x.shape), mr(mx.shape);
    r.zeros(); // it should be random
    mul(r, mac_key, mr);
    ShareTensor<T> ro_share(x.shape);
    ro_share.zeros(); // it should be random
    Tensor<T> ro(x.shape);
    restore(ro_share, ro, true);
    ShareTensor<T> v(x.shape), w(x.shape);
    mulConstAddBias(x, ro, r, v);
    mulConstAddBias(mx, ro, mr, w);
    Tensor<T> v_real(x.shape);
    restore(v, v_real, true);
    ShareTensor<T> delta(x.shape);
    mulConstSubBias(mac_key, v_real, w, delta);
    checkZero(delta);
#endif
}

template <typename T>
void twopc_protocols::macCheckSimulate(uint32_t size)
{
#if (IS_MALICIOUS)
    if (size == 0)
    {
        return;
    }
    ShareTensor<T> x({size}), mx({size}), mac_key({size});
    x.zeros();
    mx.zeros();
    mac_key.zeros();

    macCheck(x, mx, mac_key);
    MAC_SIZE = 0;
#endif
}

// 基于当前 2PC 参数结构计算最高位共享结果。
// 严格按公式逐项计算（模2）：
// DReLU_n^{[r_in, r_out]}(x_hat) = MSB_n(x_hat) ⊕ MSB_n(2^n-r_in)
//                                ⊕ 1{2^{n-1}-y0-1 < y1} ⊕ 1 ⊕ r_out
// 其中 y0 = x_hat mod 2^{n-1}, y1 = (2^n-r_in) mod 2^{n-1}。
// 字段映射（pc_cmp_param）：
// - r_in 本地份额: self_r0
// - r_in 对端份额(本地持有): r_with_next
// - r_out 本地布尔份额: self_r1
// - 比较项 DPF key: keys[0]
// - r_with_pre/r_with_pre_bits/r_with_next_bits/round*_r/round*_real_table_*:
//   保留给 privateCompare 风格路径，当前 DPF 比较路径不直接使用。
// 最后把两方布尔共享通过 b2a 转为两方算术共享返回。
template <typename T>
void twopc_protocols::pc_msb(ShareTensor<T> &x, ShareTensor<T> &res,
                           Parameters<T> &parameter, const uint32_t size, bool malicious, const char *perf_scope_prefix, uint32_t eval_bitlength)
{
    std::string perf_name_storage;
    const char *perf_name = "twopc_protocols::pc_msb";
    if (perf_scope_prefix != nullptr)
    {
        perf_name_storage = std::string(perf_scope_prefix) + "::pc_msb";
        perf_name = perf_name_storage.c_str();
    }
    perf_stats::Scope perf_scope(perf_name);
    Party2PC &party = Party2PC::getInstance();
    always_assert(size <= x.size());
    always_assert(size <= res.size());
    always_assert(party.peer_with_next != nullptr);

    const pc_cmp_param<T> &pc_cmp = parameter.select_pc_cmp(eval_bitlength);
    const uint32_t bitlength = pc_cmp.eval_bitlength;
    const T max_pos = (bitlength == static_cast<uint32_t>(sizeof(T) * 8))
                          ? static_cast<T>(static_cast<T>(-1) >> 1)
                          : static_cast<T>((T(1) << (bitlength - 1)) - 1);
    const T self_r = (party.party_id == 0) ? pc_cmp.self_r0 : pc_cmp.self_r1;

    ShareTensor<T> x_hat(x.shape);
    Tensor<T> x_open(x.shape);
    Tensor<T> dpf_bits(x.shape);
    Tensor<T> bool_res(x.shape);

    {
        std::string stage_name_storage = perf_name_storage.empty() ? std::string(perf_name) + "::form_x_hat"
                                                                   : perf_name_storage + "::form_x_hat";
        perf_stats::Scope stage_scope(stage_name_storage.c_str());
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(x.size()); ++i)
        {
            x_hat.share.data[i] = x.share.data[i] + self_r;
        }
    }

    {
        std::string stage_name_storage = perf_name_storage.empty() ? std::string(perf_name) + "::open_x_hat"
                                                                   : perf_name_storage + "::open_x_hat";
        perf_stats::Scope stage_scope(stage_name_storage.c_str());
        restore(x_hat, x_open, malicious);
    }

    {
        std::string stage_name_storage = perf_name_storage.empty() ? std::string(perf_name) + "::eval_groto_dpf"
                                                                   : perf_name_storage + "::eval_groto_dpf";
        perf_stats::Scope stage_scope(stage_name_storage.c_str());
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(size); ++i)
        {
            const T x_hat_val = x_open.data[i];
            const T ucmp_input = static_cast<T>(x_hat_val & max_pos);
            GroupElement res_elem(0, 1);
            GroupElement x_elem(static_cast<uint64_t>(ucmp_input), bitlength);
            groto_dpf::evalGrotoDPF(party.party_id, &res_elem, x_elem, pc_cmp.keys[0], false);
            dpf_bits.data[i] = static_cast<T>(res_elem.value & 1ULL);
        }
    }

    {
        std::string stage_name_storage = perf_name_storage.empty() ? std::string(perf_name) + "::form_bool_res"
                                                                   : perf_name_storage + "::form_bool_res";
        perf_stats::Scope stage_scope(stage_name_storage.c_str());
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(size); ++i)
        {
            const T x_hat_val = x_open.data[i];
            const T msb = static_cast<T>((x_hat_val >> (bitlength - 1)) & 1);
            const T dpf_bit = dpf_bits.data[i];
            if (party.party_id == 0)
            {
                bool_res.data[i] = static_cast<T>((dpf_bit + msb) & 1) ^ (pc_cmp.self_r0 & 1ULL);
            }
            else
            {
                bool_res.data[i] = dpf_bit ^ (pc_cmp.self_r1 & 1ULL);
            }
        }
        for (uint32_t i = size; i < bool_res.size(); ++i)
        {
            bool_res.data[i] = 0;
        }
    }

    // bool_res currently holds 2PC boolean shares (mod-2). Convert them to arithmetic shares.
    ass_protocols::b2a(bool_res, *(party.peer_with_next), party.party_id, perf_name);
    reshare(bool_res, res);
}

template <typename T>
void twopc_protocols::nonNegative(ShareTensor<T> &x, ShareTensor<T> &res, Parameters<T> &parameter, bool isFloat, bool malicious, const char *perf_scope_prefix, uint32_t eval_bitlength)
{
    uint32_t size = x.size();
    // pc_msb returns the sign bit (MSB): 1 if x >= 0, 0 if x < 0
    pc_msb(x, res, parameter, size, malicious, perf_scope_prefix, eval_bitlength);
    // nonNegative = 1 - MSB
    sub((T)1, res, res);
    if (isFloat)
    {
        mulConst(res, (T)(1 << x.float_scale_bit), res);
    }
}

template <typename T>
void twopc_protocols::greaterEqual(ShareTensor<T> &x, ShareTensor<T> &y, ShareTensor<T> &res, Parameters<T> &parameter, bool isFloat, bool malicious)
{
    ShareTensor<T> z(x.shape);
    sub(x, y, z);
    nonNegative(z, res, parameter, isFloat, malicious);
}

template <typename T>
void twopc_protocols::select(ShareTensor<T> &x, ShareTensor<T> &y, ShareTensor<T> &res, Parameters<T> &parameter, bool malicious, const char *perf_scope_prefix, uint32_t eval_bitlength)
{
    // 1. Get unscaled mask (0 or 1)
    // isFloat = false ensures we get pure 0/1 shares, not scaled by 2^f
    ShareTensor<T> mask(x.shape);
    nonNegative(x, mask, parameter, false, malicious, perf_scope_prefix, eval_bitlength);
    
    // 2. Multiply y by mask
    // Since mask is unscaled (0/1), the result has the same scale as y.
    // No truncation is needed (needTrunc = false).
    mul(y, mask, res, false, malicious);
}

template <typename T>
void twopc_protocols::select(ShareTensor<T> &x, ShareTensor<T> &y, ShareTensor<T> &res, uint32_t y_num, Parameters<T> &parameter, bool malicious, const char *perf_scope_prefix, uint32_t eval_bitlength)
{
    uint32_t size = x.size();
    ShareTensor<T> mask(x.shape);
    // Get unscaled mask (0 or 1)
    nonNegative(x, mask, parameter, false, malicious, perf_scope_prefix, eval_bitlength);

    ShareTensor<T> expanded_mask(y.shape);
#pragma omp parallel for
    for(uint32_t i = 0; i < y_num; ++i) {
        for(uint32_t j = 0; j < size; ++j) {
            expanded_mask.share.data[i * size + j] = mask.share.data[j];
        }
    }
    // Multiply y by expanded mask (unscaled). No truncation needed.
    mul(y, expanded_mask, res, false, malicious);
}

template <typename T>
void twopc_protocols::lut(ShareTensor<T> &x, ShareTensor<T> &res, LUT_Param<T> &parameter, bool malicious)
{
    Party2PC &party = Party2PC::getInstance();
    uint32_t size = x.size(), table_size = parameter.table_size;
    // 1. Compute x_hat = x + r
    ShareTensor<T> x_hat(x.shape);
    // Add r (RSS) to x (RSS). r is stored in parameter.r_share

    for(uint32_t i=0; i<size; ++i) {
        x_hat.share.data[i] = x.share.data[i] - parameter.self_r0;
    }

    // 2. Open x_hat
    Tensor<T> x_hat_open(x.shape);
    restore(x_hat, x_hat_open, malicious);

    // 3. Evaluation (online): compute <u> = <v> >> x_hat and w = sum_j T[j] * <u[j]>
    res.assert_same_shape(x);

    for (uint32_t i = 0; i < size; ++i)
    {
        // reduce x_hat to table domain
        uint64_t shift = (uint64_t)(x_hat_open.data[i]) % (uint64_t)table_size;

        T acc_first = 0;
        // accumulate over table entries
        for (uint32_t j = 0; j < table_size; ++j)
        {
            uint32_t idx = (uint32_t)((j + shift) % table_size);
            // v-share for this party
            T v_share_curr = parameter.onehot_table.data[idx];
            T tbl_val = parameter.table.data[j];

            acc_first += v_share_curr * tbl_val;
        }
        res.share.data[i] = acc_first;
    }


}

template <typename T>
void twopc_protocols::lut(ShareTensor<T> &x, ShareTensor<T> &res1, ShareTensor<T> &res2, LUT_Param<T> &parameter1,
                          LUT_Param<T> &parameter2, bool malicious)
{
    // TODO: extend to multi-table
    Party2PC &party = Party2PC::getInstance();
    uint32_t size = x.size();
    uint32_t combined_size = 2 * size;
    
    // 1. Compute x_hat_combined = [x - r1, x - r2]
    ShareTensor<T> x_hat_combined({combined_size});

    for(uint32_t i=0; i<size; ++i) {
        // First half: x - r1 (for parameter1)
        x_hat_combined.share.data[i] = x.share.data[i] - parameter1.self_r1;
        
        // Second half: x - r2 (for parameter2)
        x_hat_combined.share.data[i + size] = x.share.data[i] - parameter2.self_r1;
    }

    // 2. Open x_hat (One communication round for both)
    Tensor<T> x_hat_open({combined_size});
    restore(x_hat_combined, x_hat_open, malicious);

    // 3. Evaluation (Online)
    res1.assert_same_shape(x);
    res2.assert_same_shape(x);


    for (uint32_t i = 0; i < size; ++i)
    {
        // --- Table 1 Evaluation ---
        uint64_t shift1 = (uint64_t)(x_hat_open.data[i]) % (uint64_t)parameter1.table_size;
        T acc1_first = 0;
        
        for (uint32_t j = 0; j < parameter1.table_size; ++j)
        {
            uint32_t idx = (uint32_t)((j + shift1) % parameter1.table_size);
            T v_curr = parameter1.onehot_table.data[idx];
            T tbl_val = parameter1.table.data[j];
            
            acc1_first += v_curr * tbl_val;
        }
        res1.share.data[i] = acc1_first;

        // --- Table 2 Evaluation ---
        uint64_t shift2 = (uint64_t)(x_hat_open.data[i + size]) % (uint64_t)parameter2.table_size;
        T acc2_first = 0;
        
        for (uint32_t j = 0; j < parameter2.table_size; ++j)
        {
            uint32_t idx = (uint32_t)((j + shift2) % parameter2.table_size);
            T v_curr = parameter2.onehot_table.data[idx];
            T tbl_val = parameter2.table.data[j];
            
            acc2_first += v_curr * tbl_val;
        }
        res2.share.data[i] = acc2_first;

    }

    
}

template <typename T>
void twopc_protocols::utils::getk(ShareTensor<T> &x, ShareTensor<T> &k, Parameters<T> &parameters, bool malicious, uint32_t eval_bitlength)
{
    // 2PC version: find k such that b^k <= x < b^(k+1), then output k+1.
    Party2PC &party = Party2PC::getInstance();
    uint32_t size = x.size();
    uint32_t nexpb_size = (int)(log(pow(2, 2 * x.float_scale_bit)) / log(SCALE_BASE));
    ShareTensor<T> delta({size, nexpb_size}); // x - b ^ i for i from 1 to max size

    if (party.party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < nexpb_size; j++)
            {
                delta.share.data[i * nexpb_size + j] = x.share.data[i] - (T)(pow(SCALE_BASE, j + 1));
            }
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < nexpb_size; j++)
            {
                delta.share.data[i * nexpb_size + j] = x.share.data[i];
            }
        }
    }
    nonNegative(delta, delta, parameters, false, malicious, nullptr, eval_bitlength);

    // calculate k + 1
    if (party.party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            k.share.data[i] = 1;
            for (int j = 0; j < nexpb_size; j++)
            {
                k.share.data[i] += delta.share.data[i * nexpb_size + j];
            }
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            k.share.data[i] = 0;
            for (int j = 0; j < nexpb_size; j++)
            {
                k.share.data[i] += delta.share.data[i * nexpb_size + j];
            }
        }
    }
}

template <typename T>
void twopc_protocols::inv(ShareTensor<T> &x, ShareTensor<T> &res, uint32_t dim_size, Parameters<T> &parameters, bool malicious)
{
    constexpr uint32_t chopped_frac_bits = 6;
    constexpr uint32_t full_frac_bits = static_cast<uint32_t>(kFloat_Precision<T>);
    static_assert(full_frac_bits >= chopped_frac_bits,
                  "inv single-LUT path requires kFloat_Precision<T> >= 6");

    // Sequence length k is passed explicitly from softmax reduction dimension.
    always_assert(dim_size > 0);
    const uint32_t seq_len_k = dim_size;

    always_assert(parameters.inv_chopped_max_k > 0);
    always_assert(seq_len_k <= parameters.inv_chopped_max_k);

    // Chopping precision: x_q = floor(x / 2^(f-6)), with scale 2^6.
    ShareTensor<T> x_q(x.shape), inv_res(x.shape);
    truncate(x, x_q, static_cast<size_t>(1ULL << (full_frac_bits - chopped_frac_bits)), malicious);

    // Single lookup returns value already scaled by 2^f: inv(x) ~= floor(2^(f+6) / x_q).
    lut(x_q, inv_res, parameters.inv_chopped_param, malicious);
    res = inv_res;
}

template <typename T>
void twopc_protocols::rsqrt(ShareTensor<T> &x, ShareTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
    constexpr uint32_t exponent_bits = 6;
    constexpr uint32_t mantissa_bits = 7;
    constexpr uint32_t bit_width = static_cast<uint32_t>(sizeof(T) * 8);
    constexpr uint32_t kRsqrtEvalBitlength = bit_width >= 62 ? 62 : bit_width;
    constexpr uint32_t mantissa_base = (1U << mantissa_bits);
    constexpr uint32_t expected_table_size = (1U << (exponent_bits + mantissa_bits));

    x.assert_same_shape(res);
    always_assert(parameters.rsqrt_param.table_size == expected_table_size);
    always_assert(parameters.rsqrt_shift_param.table_size == bit_width + 1);

    // Step 1: secure interval lookup to get bit-length k.
    ShareTensor<T> k(x.shape), e(x.shape);
    utils::getk(x, k, parameters, malicious, kRsqrtEvalBitlength);
    sub(k, static_cast<T>(1), e); // e = k - 1

    // Step 2/3: LUT returns u = 2^(n-k), then t = x * u.
    ShareTensor<T> u(x.shape), t(x.shape);
    lut(k, u, parameters.rsqrt_shift_param, malicious);
    mul(x, u, t, true, malicious);

    // Step 4: m = floor(t / 2^(n-8)) - 128.
    ShareTensor<T> m(x.shape);
    truncate(t, m, static_cast<size_t>(1ULL << (bit_width - mantissa_bits)), malicious);
    sub(m, static_cast<T>(mantissa_base), m);

    // Step 5/6: p = m * 2^6 + e, then LUT for rsqrt.
    ShareTensor<T> index(x.shape);
    mulConst(m, static_cast<T>(1U << exponent_bits), index);
    add(index, e, index);
    lut(index, res, parameters.rsqrt_param, malicious);
}

template <typename T>
void twopc_protocols::utils::gelu_same_scale(ShareTensor<T> &x, ShareTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
    constexpr uint32_t bit_width = static_cast<uint32_t>(sizeof(T) * 8);
    constexpr uint32_t kGeluTableBitlength = bit_width > 47 ? 47 : bit_width;
    T table_size = parameters.gelu_param.table_size;
    uint32_t size = x.size();
    ShareTensor<T> y(x.shape);

    ShareTensor<T> dx(x.shape); // dx = relu(x)
    {
        perf_stats::Scope stage_scope("twopc_protocols::gelu::select_relu");
        twopc_protocols::select(x, x, dx, parameters, malicious, "twopc_protocols::gelu::select_relu", kGeluTableBitlength);
    }

    ShareTensor<T> abs_x(x.shape), sizeSubAbs(x.shape);
    twopc_protocols::mulConstSubBias(dx, (T)2, x, abs_x); // calculate abs_x = dx * 2 - x

    twopc_protocols::sub(table_size, abs_x, sizeSubAbs);
    ShareTensor<T> ia(x.shape);
    {
        perf_stats::Scope stage_scope("twopc_protocols::gelu::select_abs");
        twopc_protocols::select(sizeSubAbs, abs_x, ia, parameters, malicious, "twopc_protocols::gelu::select_abs", kGeluTableBitlength);
    }
    twopc_protocols::add(ia, (T)(table_size - 1), ia);
    ShareTensor<T> c(x.shape);
    {
        perf_stats::Scope stage_scope("twopc_protocols::gelu::lut");
        twopc_protocols::lut(ia, c, parameters.gelu_param, malicious);
    }
    twopc_protocols::sub(dx, c, res);
}

template <typename T>
void twopc_protocols::gelu(ShareTensor<T> &x, ShareTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
    perf_stats::Scope perf_scope("twopc_protocols::gelu");
    constexpr uint32_t bit_width = static_cast<uint32_t>(sizeof(T) * 8);
    constexpr uint32_t kGeluInputBitlength = bit_width >= 62 ? 62 : bit_width;
    constexpr uint32_t kGeluTableBitlength = bit_width > 47 ? 47 : bit_width;
     if (x.float_scale_bit == GELU_TABLE_PRECISION)
    {
        utils::gelu_same_scale(x, res, parameters, malicious);
    }
    else
    {
        T table_size = parameters.gelu_param.table_size;
        uint32_t size = x.size();
        ShareTensor<T> y(x.shape);
        {
            perf_stats::Scope stage_scope("twopc_protocols::gelu::truncate");
            truncate(x, y, 1 << (x.float_scale_bit - GELU_TABLE_PRECISION), malicious);
        }

        ShareTensor<T> dx(x.shape), dy(x.shape);
        {
            perf_stats::Scope stage_scope("twopc_protocols::gelu::select_relu");
            select(x, x, dx, parameters, malicious, "twopc_protocols::gelu::select_relu", kGeluInputBitlength);
            select(y, y, dy, parameters, malicious, "twopc_protocols::gelu::select_relu_truncated", kGeluTableBitlength);
        }

        ShareTensor<T> abs_y(x.shape), sizeSubAbs(x.shape);
        mulConstSubBias(dy, (T)2, y, abs_y); // calculate abs_y = dy * 2 - y
        sub(table_size, abs_y, sizeSubAbs);
        ShareTensor<T> ia(x.shape);
        {
            perf_stats::Scope stage_scope("twopc_protocols::gelu::select_abs");
            select(sizeSubAbs, abs_y, ia, parameters, malicious, "twopc_protocols::gelu::select_abs", kGeluTableBitlength);
        }
        add(ia, (T)(table_size - 1), ia);
        ShareTensor<T> c(x.shape);
        {
            perf_stats::Scope stage_scope("twopc_protocols::gelu::lut");
            lut(ia, c, parameters.gelu_param, malicious);
        }
        sub(dx, c, res);
    }
}

template <typename T>
void twopc_protocols::silu(ShareTensor<T> &x, ShareTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
    constexpr uint32_t bit_width = static_cast<uint32_t>(sizeof(T) * 8);
    constexpr uint32_t kSiluInputBitlength = bit_width >= 62 ? 62 : bit_width;
    constexpr uint32_t kSiluTableBitlength = bit_width > 47 ? 47 : bit_width;
    if (x.float_scale_bit == SWIGLU_TABLE_PRECISION)
    {
        const T table_size = parameters.swiglu_param.table_size;
        ShareTensor<T> relu_x(x.shape), abs_x(x.shape), size_sub_abs(x.shape), ia(x.shape), delta(x.shape);
        select(x, x, relu_x, parameters, malicious, nullptr, kSiluTableBitlength);
        mulConstSubBias(relu_x, static_cast<T>(2), x, abs_x);
        sub(table_size, abs_x, size_sub_abs);
        select(size_sub_abs, abs_x, ia, parameters, malicious, nullptr, kSiluTableBitlength);
        add(ia, static_cast<T>(table_size - 1), ia);
        lut(ia, delta, parameters.swiglu_param, malicious);
        sub(relu_x, delta, res);
        return;
    }

    const T table_size = parameters.swiglu_param.table_size;
    const uint32_t size = x.size();
    ShareTensor<T> x_scaled(x.shape);
    truncate(x, x_scaled, static_cast<size_t>(1ULL << (x.float_scale_bit - SWIGLU_TABLE_PRECISION)), malicious);
    ShareTensor<T> relu_x(x.shape), relu_scaled(x.shape);
    select(x, x, relu_x, parameters, malicious, nullptr, kSiluInputBitlength);
    select(x_scaled, x_scaled, relu_scaled, parameters, malicious, nullptr, kSiluTableBitlength);

    ShareTensor<T> abs_scaled(x.shape), size_sub_abs(x.shape), ia(x.shape), delta(x.shape);
    mulConstSubBias(relu_scaled, static_cast<T>(2), x_scaled, abs_scaled);
    sub(table_size, abs_scaled, size_sub_abs);
    select(size_sub_abs, abs_scaled, ia, parameters, malicious, nullptr, kSiluTableBitlength);
    add(ia, static_cast<T>(table_size - 1), ia);
    lut(ia, delta, parameters.swiglu_param, malicious);
    sub(relu_x, delta, res);
}

template <typename T>
void twopc_protocols::swiglu(ShareTensor<T> &gate, ShareTensor<T> &up, ShareTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
    perf_stats::Scope perf_scope("twopc_protocols::swiglu");
    ShareTensor<T> silu_gate(gate.shape);
    silu(gate, silu_gate, parameters, malicious);
    mul(silu_gate, up, res, true, malicious);
}

template <typename T>
void twopc_protocols::rms_norm(ShareTensor<T> &x, ShareTensor<T> &weight, ShareTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
    perf_stats::Scope perf_scope("twopc_protocols::rms_norm");
    std::vector<uint32_t> sum_shape = x.shape;
    const uint32_t dim_size = sum_shape.back();
    always_assert(dim_size == weight.shape[0]);
    sum_shape.pop_back();

    ShareTensor<T> x_square(x.shape), square_sum(sum_shape);
    square(x, x_square, true, malicious);

    const uint32_t sum_size = square_sum.size();
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(sum_size); i++)
    {
        square_sum.share.data[i] = 0;
        for (int j = 0; j < static_cast<int>(dim_size); j++)
        {
            square_sum.share.data[i] += x_square.share.data[i * dim_size + j];
        }
    }

    truncate(square_sum, dim_size, malicious);
    add(square_sum, static_cast<T>(1), square_sum);

    ShareTensor<T> inv_rms(sum_shape), broadcast_inv_rms(x.shape), gamma(x.shape), normalized(x.shape);
    rsqrt(square_sum, inv_rms, parameters, malicious);

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(sum_size); i++)
    {
        for (int j = 0; j < static_cast<int>(dim_size); j++)
        {
            broadcast_inv_rms.share.data[i * dim_size + j] = inv_rms.share.data[i];
            gamma.share.data[i * dim_size + j] = weight.share.data[j];
        }
    }

    mul(x, broadcast_inv_rms, normalized, true, malicious);
    mul(normalized, gamma, res, true, malicious);
}

template <typename T>
void twopc_protocols::max_last_dim(ShareTensor<T> &x, ShareTensor<T> &res, Parameters<T> &parameter, bool malicious)
{
    constexpr uint32_t bit_width = static_cast<uint32_t>(sizeof(T) * 8);
    constexpr uint32_t kMaxLastDimEvalBitlength = bit_width > 52 ? 52 : bit_width;
    uint32_t last_dim_size = x.shape.back();
    ShareTensor<T> tmp(x), even, odd, delta;
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
                even.share.data[index0] = tmp.share.data[index1];
                odd.share.data[index0] = tmp.share.data[index1 + 1];

            }
            index0 = i * half_size + half_size - 1;
            index1 = i * last_dim_size + 2 * (half_size - 1);
            even.share.data[index0] = tmp.share.data[index1];

            odd.share.data[index0] = tmp.share.data[i * last_dim_size + last_dim_size - 1];
        }

        sub(even, odd, delta);
    select(delta, delta, delta, parameter, malicious, nullptr, kMaxLastDimEvalBitlength);
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
void twopc_protocols::neg_exp(ShareTensor<T> &x, ShareTensor<T> &res, Parameters<T> &parameter, bool malicious)
{
    constexpr int target_frac_bits = 12;
    constexpr int input_frac_bits = kFloat_Precision<T>;
    constexpr uint32_t kNegExpEvalBitlength = (sizeof(T) * 8) > 52 ? 52 : static_cast<uint32_t>(sizeof(T) * 8);

    // 1) Convert input scale to f=12 domain.
    ShareTensor<T> x12(x.shape);
    if constexpr (input_frac_bits > target_frac_bits)
    {
        truncate(x, x12, static_cast<size_t>(1ULL << (input_frac_bits - target_frac_bits)), malicious);
    }
    else if constexpr (input_frac_bits < target_frac_bits)
    {
        mulConst(x, static_cast<T>(1ULL << (target_frac_bits - input_frac_bits)), x12);
    }
    else
    {
        x12 = x;
    }

    // 2) Clip x to [0, 16) under f=12.
    // Input is guaranteed non-negative, so only upper-bound clipping is needed.

    const int64_t clip_hi_signed = (static_cast<int64_t>(16) << target_frac_bits) - 1;
    const T clip_hi = detail::signed_to_ring<T>(clip_hi_signed);
    ShareTensor<T> margin(x.shape), in_range_mask(x.shape), over_mask(x.shape);
    twopc_protocols::sub(clip_hi, x12, margin);
    twopc_protocols::nonNegative(margin, in_range_mask, parameter, false, malicious, nullptr, kNegExpEvalBitlength); // 1 if x12 <= clip_hi else 0
    twopc_protocols::sub(static_cast<T>(1), in_range_mask, over_mask);

    ShareTensor<T> part_in_range(x.shape), part_over(x.shape), clipped12(x.shape);
    twopc_protocols::mul(x12, in_range_mask, part_in_range, false, malicious);
    twopc_protocols::mulConst(over_mask, clip_hi, part_over);
    twopc_protocols::add(part_in_range, part_over, clipped12);

    // 3) Split 16-bit fixed-point c into c1 (high 8 bits) and c0 (low 8 bits).
    ShareTensor<T> c1(x.shape), c1_scaled(x.shape), c0(x.shape);
    twopc_protocols::truncate(clipped12, c1, static_cast<size_t>(256), malicious); // c1 = floor(c / 256)
    twopc_protocols::mulConst(c1, static_cast<T>(256), c1_scaled);
    twopc_protocols::sub(clipped12, c1_scaled, c0); // c0 = c - c1 * 256

    // 4) Lookup and combine: exp(-c) = exp(-c1/16) * exp(-c0/4096).
    ShareTensor<T> y_hi(x.shape), y_lo(x.shape), prod(x.shape), y12(x.shape);
    twopc_protocols::lut(c1, y_hi, parameter.exp_hi_param, malicious);
    twopc_protocols::lut(c0, y_lo, parameter.exp_lo_param, malicious);
    twopc_protocols::mul(y_hi, y_lo, prod, false, malicious);
    twopc_protocols::truncate(prod, y12, static_cast<size_t>(1ULL << target_frac_bits), malicious);

    // 5) Convert result back to input/output fixed-point scale.
    if constexpr (input_frac_bits > target_frac_bits)
    {
        twopc_protocols::mulConst(y12, static_cast<T>(1ULL << (input_frac_bits - target_frac_bits)), res);
    }
    else if constexpr (input_frac_bits < target_frac_bits)
    {
        twopc_protocols::truncate(y12, res, static_cast<size_t>(1ULL << (target_frac_bits - input_frac_bits)), malicious);
    }
    else
    {
        res = y12;
    }
}

template <typename T>
void twopc_protocols::softmax_forward(ShareTensor<T> &x, ShareTensor<T> &res, Parameters<T> &parameter, bool malicious)
{
    perf_stats::Scope perf_scope("twopc_protocols::softmax_forward");
    const auto profile_begin = std::chrono::steady_clock::now();

    /* only support dim = -1*/
    std::vector<uint32_t> sum_shape = x.shape;
    uint32_t dim_size = sum_shape.back();
    sum_shape.pop_back();

    ShareTensor<T> x_max(sum_shape), delta(x.shape);
    uint32_t common_size = x_max.size();
    max_last_dim(x, x_max, parameter, malicious);

    for (int i = 0; i < common_size; i++)
    {
        for (int j = 0; j < dim_size; j++)
        {
            const int index = i * dim_size + j;
            delta.share.data[index] = x_max.share.data[i] - x.share.data[index];
        }
    }

    ShareTensor<T> exp_x(x.shape);
    neg_exp(delta, exp_x, parameter, malicious);
    ShareTensor<T> sum(sum_shape);

#pragma omp parallel for
    for (int i = 0; i < common_size; i++)
    {
        sum.share.data[i] = 0;
        for (int j = 0; j < dim_size; j++)
        {
            sum.share.data[i] += exp_x.share.data[i * dim_size + j];
        }
    }

    inv(sum, sum, dim_size, parameter, malicious);
    ShareTensor<T> broadcast_sum(x.shape);
    for (int i = 0; i < common_size; i++)
    {
        for (int j = 0; j < dim_size; j++)
        {
            broadcast_sum.share.data[i * dim_size + j] = sum.share.data[i];
        }
    }
    mul(exp_x, broadcast_sum, res, true, malicious);

    const uint64_t elapsed_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - profile_begin)
            .count());
    twopc_protocols::profile::add_softmax_ns(elapsed_ns);
}

template <typename T>
void twopc_protocols::grouped_multi_query_attention(ShareTensor<T> &query, ShareTensor<T> &key, ShareTensor<T> &value, ShareTensor<T> &res,
                                                    Parameters<T> &parameter, uint32_t num_heads,
                                                    uint32_t num_key_value_heads, bool masked, bool malicious)
{
    perf_stats::Scope perf_scope("twopc_protocols::grouped_multi_query_attention");
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

    ShareTensor<T> query_heads = query;
    query_heads.reshape({batch_size, seq_len, num_heads, head_size});
    ShareTensor<T> key_heads = key;
    key_heads.reshape({batch_size, seq_len, num_key_value_heads, head_size});
    ShareTensor<T> value_heads = value;
    value_heads.reshape({batch_size, seq_len, num_key_value_heads, head_size});

    ShareTensor<T> query_layer = query_heads.permute({0, 2, 1, 3});
    ShareTensor<T> key_layer = key_heads.permute({0, 2, 3, 1});
    ShareTensor<T> value_layer = value_heads.permute({0, 2, 1, 3});

    ShareTensor<T> key_for_attn;
    ShareTensor<T> value_for_attn;
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
                        key_for_attn.share.data[key_dst] = key_layer.share.data[key_src];

                        const uint32_t value_dst = ((static_cast<uint32_t>(b) * num_heads + h) * seq_len + s) * head_size + d;
                        const uint32_t value_src = ((static_cast<uint32_t>(b) * num_key_value_heads + kv_head) * seq_len + s) * head_size + d;
                        value_for_attn.share.data[value_dst] = value_layer.share.data[value_src];
                    }
                }
            }
        }
    }

    ShareTensor<T> attention_scores({batch_size, num_heads, seq_len, seq_len});
    {
        perf_stats::Scope qk_scope("twopc_protocols::grouped_multi_query_attention::qk_matmul");
        twopc_protocols::matMul(query_layer, key_for_attn, attention_scores, true, malicious);
    }
    twopc_protocols::truncate(attention_scores, static_cast<uint32_t>(std::sqrt(static_cast<double>(head_size))), malicious);

    if (masked)
    {
        const uint32_t common_size = batch_size * num_heads;
#pragma omp parallel for if (parallel::should_parallelize(static_cast<uint64_t>(common_size) * seq_len * seq_len))
        for (int i = 0; i < static_cast<int>(common_size); i++)
        {
            for (uint32_t q = 0; q < seq_len; q++)
            {
                const uint32_t row_base = (static_cast<uint32_t>(i) * seq_len + q) * seq_len;
                const T fill = attention_scores.share.data[row_base];
                for (uint32_t k = q + 1; k < seq_len; k++)
                {
                    attention_scores.share.data[row_base + k] = fill;
                }
            }
        }
    }

    ShareTensor<T> attention_probs(attention_scores.shape);
    twopc_protocols::softmax_forward(attention_scores, attention_probs, parameter, malicious);

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
                    attention_probs.share.data[row_base + k] = 0;
                }
            }
        }
    }

    ShareTensor<T> context_layer({batch_size, num_heads, seq_len, head_size});
    {
        perf_stats::Scope av_scope("twopc_protocols::grouped_multi_query_attention::av_matmul");
        twopc_protocols::matMul(attention_probs, value_for_attn, context_layer, true, malicious);
    }

    ShareTensor<T> attn_output = context_layer.permute({0, 2, 1, 3});
    attn_output.reshape({batch_size, seq_len, hidden_size});
    res = attn_output;
}

