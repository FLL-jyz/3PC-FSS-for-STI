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

        template <typename T>
        inline double ring_to_double_fixed(const T v, uint32_t frac_bits)
        {
            const double scale = static_cast<double>(1ULL << frac_bits);
            return static_cast<double>(ring_to_signed(v)) / scale;
        }

        // Encode a double into the fixed-point representation used by T.
        template <typename T>
        inline T fixed_to_ring(const double v)
        {
            const double scale = static_cast<double>(1ULL << kFloat_Precision<T>);
            return double_to_ring<T>(v * scale);
        }

        template <typename T>
        inline T fixed_to_ring(const double v, uint32_t frac_bits)
        {
            const double scale = static_cast<double>(1ULL << frac_bits);
            return double_to_ring<T>(v * scale);
        }

        // Resolve the signed interpretation width used by A2B-based sign extraction.
        template <typename T>
        inline uint32_t resolve_signed_bitlength(uint32_t eval_bitlength)
        {
            return eval_bitlength != 0 ? eval_bitlength : static_cast<uint32_t>(sizeof(T) * 8);
        }

        template <typename T>
        inline uint32_t softmax_eval_bitlength()
        {
            constexpr uint32_t bit_width = static_cast<uint32_t>(sizeof(T) * 8);
            return bit_width > 52 ? 52 : bit_width;
        }

        // Clip the softmax input to [-4, 12] and fold the public 1/16 factor into the tensor.
        template <typename T>
        inline void prepare_softmax_input(ShareTensor<T> &x, ShareTensor<T> &x_scaled, Parameters<T> &parameter, bool malicious, const char *perf_scope_prefix = nullptr)
        {
            const T clip_lo = fixed_to_ring<T>(-4.0);
            const T clip_hi = fixed_to_ring<T>(12.0);

            ShareTensor<T> lower_margin(x.shape), upper_margin(x.shape);
            ShareTensor<T> above_lower_mask(x.shape), below_upper_mask(x.shape), in_range_mask(x.shape);
            ShareTensor<T> below_mask(x.shape), above_mask(x.shape);
            ShareTensor<T> clipped_mid(x.shape), clipped_low(x.shape), clipped_high(x.shape), clipped_x(x.shape);

            std::string prefix = perf_scope_prefix == nullptr ? std::string("twopc_protocols::softmax_prepare")
                                                              : std::string(perf_scope_prefix);

            sub(x, clip_lo, lower_margin);
            nonNegative(lower_margin, above_lower_mask, parameter, false, malicious, (prefix + "::clip_lower").c_str(), softmax_eval_bitlength<T>());
            sub(clip_hi, x, upper_margin);
            nonNegative(upper_margin, below_upper_mask, parameter, false, malicious, (prefix + "::clip_upper").c_str(), softmax_eval_bitlength<T>());
            mul(above_lower_mask, below_upper_mask, in_range_mask, false, malicious);
            sub(static_cast<T>(1), above_lower_mask, below_mask);
            sub(static_cast<T>(1), below_upper_mask, above_mask);

            mul(x, in_range_mask, clipped_mid, false, malicious);
            mulConst(below_mask, clip_lo, clipped_low);
            mulConst(above_mask, clip_hi, clipped_high);
            add(clipped_mid, clipped_low, clipped_x);
            add(clipped_x, clipped_high, clipped_x);
            truncate(clipped_x, x_scaled, static_cast<size_t>(16), malicious);
        }

        template <typename T>
        inline void exp_limit(ShareTensor<T> &x, ShareTensor<T> &res, bool malicious,
                      uint32_t squaring_rounds = 8);

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

        // Build the Fourier residual delta(|x|) with secure sine evaluations on the clipped |x| domain.
        template <typename T>
        void gelu_fourier_delta(ShareTensor<T> &abs_x, ShareTensor<T> &delta,
                    Parameters<T> &parameters, bool malicious = false,
                    const char *perf_scope_prefix = nullptr,
                    uint32_t input_frac_bits = kFloat_Precision<T>,
                    uint32_t eval_bitlength = 0);

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
                const uint32_t size, bool malicious = false, const char *perf_scope_prefix = nullptr, uint32_t eval_bitlength = 0);

    template <typename T>
    void nonNegative(ShareTensor<T> &x, ShareTensor<T> &res, Parameters<T> &parameter, bool isFloat = true, bool malicious = false, const char *perf_scope_prefix = nullptr, uint32_t eval_bitlength = 0);

    template <typename T>
    void select(ShareTensor<T> &x, ShareTensor<T> &y, ShareTensor<T> &res, Parameters<T> &parameter, bool malicious = false, const char *perf_scope_prefix = nullptr, uint32_t eval_bitlength = 0);

    template <typename T>
    void select(ShareTensor<T> &x, ShareTensor<T> &y, ShareTensor<T> &res, uint32_t y_num, Parameters<T> &parameter, bool malicious = false, const char *perf_scope_prefix = nullptr, uint32_t eval_bitlength = 0);

    template <typename T>
    void rsqrt(ShareTensor<T> &x, ShareTensor<T> &res, Parameters<T> &parameters, bool malicious = false);

    template <typename T>
    void secure_sin(ShareTensor<T> &x, ShareTensor<T> &res, bool malicious = false,
                    const char *perf_scope_prefix = nullptr,
                    uint32_t input_frac_bits = kFloat_Precision<T>);

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

namespace twopc_protocols::detail
{
    template <typename T>
    inline void exp_limit(ShareTensor<T> &x, ShareTensor<T> &res, bool malicious,
                          uint32_t squaring_rounds)
    {
        ShareTensor<T> scaled(x.shape), current(x.shape), next(x.shape);
        twopc_protocols::truncate(x, scaled, static_cast<size_t>(1ULL << squaring_rounds), malicious);
        twopc_protocols::add(scaled, fixed_to_ring<T>(1.0, kFloat_Precision<T>), current);

        for (uint32_t round = 0; round < squaring_rounds; ++round)
        {
            twopc_protocols::square(current, next, true, malicious);
            current = next;
        }
        res = current;
    }
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

// 先把两方算术共享转换成 packed boolean share，提取解释位宽下的符号位布尔共享，
// 再通过 b2a 把该单比特布尔共享提升回算术共享。
template <typename T>
void twopc_protocols::pc_msb(ShareTensor<T> &x, ShareTensor<T> &res,
                           const uint32_t size, bool malicious, const char *perf_scope_prefix, uint32_t eval_bitlength)
{
    parallel::ScopedOmpThreadLimit scoped_thread_limit(8);
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
    (void)malicious;

    const uint32_t bitlength = detail::resolve_signed_bitlength<T>(eval_bitlength);
    always_assert(bitlength > 0);
    always_assert(bitlength <= static_cast<uint32_t>(sizeof(T) * 8));

    Tensor<T> bool_res(x.shape);
    bool_res.zero();
    Tensor<T> bool_share({size});

    {
        std::string stage_name_storage = perf_name_storage.empty() ? std::string(perf_name) + "::a2b"
                                                                   : perf_name_storage + "::a2b";
        perf_stats::Scope stage_scope(stage_name_storage.c_str());
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(size); ++i)
        {
            bool_share.data[i] = x.share.data[i];
        }
        ass_protocols::a2b(bool_share, *(party.peer_with_next), party.party_id, perf_name, 1024, bitlength);
    }

    {
        std::string stage_name_storage = perf_name_storage.empty() ? std::string(perf_name) + "::extract_msb"
                                                                   : perf_name_storage + "::extract_msb";
        perf_stats::Scope stage_scope(stage_name_storage.c_str());
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(size); ++i)
        {
            bool_res.data[i] = static_cast<T>((bool_share.data[i] >> (bitlength - 1)) & 1);
        }
    }

    // bool_res currently holds the extracted MSB as 2PC boolean shares (mod-2).
    ass_protocols::b2a(bool_res, *(party.peer_with_next), party.party_id, perf_name);
    reshare(bool_res, res);
}

template <typename T>
void twopc_protocols::nonNegative(ShareTensor<T> &x, ShareTensor<T> &res, Parameters<T> &parameter, bool isFloat, bool malicious, const char *perf_scope_prefix, uint32_t eval_bitlength)
{
    uint32_t size = x.size();
    (void)parameter;
    // pc_msb returns the arithmetic sharing of the sign bit: 1 if x < 0, 0 otherwise.
    pc_msb(x, res, size, malicious, perf_scope_prefix, eval_bitlength);
    // nonNegative = 1 - MSB
    sub((T)1, res, res);
    if (isFloat)
    {
        mulConst(res, (T)(1 << x.float_scale_bit), res);
    }
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
void twopc_protocols::rsqrt(ShareTensor<T> &x, ShareTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
    (void)parameters;
    constexpr uint32_t kRsqrtIterations = 10;

    ShareTensor<T> exponent_arg(x.shape), guess(x.shape), y_sq(x.shape), xy_sq(x.shape), correction(x.shape), next(x.shape);
    sub(detail::fixed_to_ring<T>(1.0, kFloat_Precision<T>), x, exponent_arg);
    truncate(exponent_arg, exponent_arg, static_cast<size_t>(2), malicious);
    detail::exp_limit(exponent_arg, guess, malicious);

    ShareTensor<T> current = guess;
    const T three = detail::fixed_to_ring<T>(3.0, kFloat_Precision<T>);
    for (uint32_t iter = 0; iter < kRsqrtIterations; ++iter)
    {
        square(current, y_sq, true, malicious);
        mul(x, y_sq, xy_sq, true, malicious);
        sub(three, xy_sq, correction);
        mul(current, correction, next, true, malicious);
        truncate(next, next, static_cast<size_t>(2), malicious);
        current = next;
    }
    res = current;
}

template <typename T>
void twopc_protocols::secure_sin(ShareTensor<T> &x, ShareTensor<T> &res, bool malicious,
                                 const char *perf_scope_prefix,
                                 uint32_t input_frac_bits)
{
    const std::string prefix = perf_scope_prefix == nullptr ? std::string("twopc_protocols::secure_sin")
                                                            : std::string(perf_scope_prefix);
    ShareTensor<T> t(x.shape), sin_t(x.shape), cos_t(x.shape), delta_share(x.shape);
    Tensor<T> delta_open(x.shape), sin_delta(x.shape), cos_delta(x.shape);
    ShareTensor<T> sin_delta_cos_t_scaled(x.shape), cos_delta_sin_t_scaled(x.shape);
    ShareTensor<T> sin_delta_cos_t(x.shape), cos_delta_sin_t(x.shape);

    {
        perf_stats::Scope stage_scope((prefix + "::load_offline").c_str());
        detail::load_offline_triple_from_file("SHAFT_OFFLINE_SIN_FILE", t.share, sin_t.share, cos_t.share);
    }

    sub(x, t, delta_share);
    {
        perf_stats::Scope stage_scope((prefix + "::open_delta").c_str());
        restore(delta_share, delta_open, malicious);
    }

#pragma omp parallel for
    for (int i = 0; i < delta_open.size(); i++)
    {
        const double real_delta = detail::ring_to_double_fixed(delta_open.data[i], input_frac_bits);
        sin_delta.data[i] = detail::fixed_to_ring<T>(std::sin(real_delta));
        cos_delta.data[i] = detail::fixed_to_ring<T>(std::cos(real_delta));
    }

    {
        perf_stats::Scope stage_scope((prefix + "::combine").c_str());
        mulConst(cos_t, sin_delta, sin_delta_cos_t_scaled);
        mulConst(sin_t, cos_delta, cos_delta_sin_t_scaled);
        truncate(sin_delta_cos_t_scaled, sin_delta_cos_t, static_cast<size_t>(1ULL << input_frac_bits), malicious);
        truncate(cos_delta_sin_t_scaled, cos_delta_sin_t, static_cast<size_t>(1ULL << input_frac_bits), malicious);
        add(sin_delta_cos_t, cos_delta_sin_t, res);
    }
}

template <typename T>
void twopc_protocols::utils::gelu_fourier_delta(ShareTensor<T> &abs_x, ShareTensor<T> &delta,
                                                Parameters<T> &parameters, bool malicious,
                                                const char *perf_scope_prefix,
                                                uint32_t input_frac_bits,
                                                uint32_t eval_bitlength)
{
    constexpr size_t kFourierTerms = 8;
    constexpr double kPi = 3.141592653589793238462643383279502884;

    const std::string prefix = perf_scope_prefix == nullptr ? std::string("twopc_protocols::gelu")
                                                            : std::string(perf_scope_prefix);
    const T domain_limit = detail::fixed_to_ring<T>(4.0, input_frac_bits);

    ShareTensor<T> margin(abs_x.shape), in_range_mask(abs_x.shape), angle_scaled(abs_x.shape), angle(abs_x.shape);
    ShareTensor<T> sine_term(abs_x.shape), weighted_term_scaled(abs_x.shape), weighted_term(abs_x.shape), masked_delta(abs_x.shape);
    Tensor<T> harmonic_tensor(abs_x.shape), beta_tensor(abs_x.shape);

    sub(domain_limit, abs_x, margin);
    {
        perf_stats::Scope stage_scope((prefix + "::in_range").c_str());
        nonNegative(margin, in_range_mask, parameters, false, malicious, (prefix + "::in_range").c_str(), eval_bitlength);
    }

    delta.zeros();
    for (size_t term_idx = 0; term_idx < kFourierTerms; ++term_idx)
    {
        const double harmonic = static_cast<double>(term_idx + 1) * (kPi / 4.0);
#pragma omp parallel for
        for (int i = 0; i < abs_x.size(); i++)
        {
            harmonic_tensor.data[i] = detail::fixed_to_ring<T>(harmonic, input_frac_bits);
            beta_tensor.data[i] = detail::fixed_to_ring<T>(parameters.gelu_fourier_beta[term_idx], input_frac_bits);
        }

        mulConst(abs_x, harmonic_tensor, angle_scaled);
        truncate(angle_scaled, angle, static_cast<size_t>(1ULL << input_frac_bits), malicious);
        secure_sin(angle, sine_term, malicious, (prefix + "::secure_sin_" + std::to_string(term_idx + 1)).c_str(), input_frac_bits);
        mulConst(sine_term, beta_tensor, weighted_term_scaled);
        truncate(weighted_term_scaled, weighted_term, static_cast<size_t>(1ULL << input_frac_bits), malicious);

        if (term_idx == 0)
        {
            delta = weighted_term;
        }
        else
        {
            add(delta, weighted_term, delta);
        }
    }

    mul(delta, in_range_mask, masked_delta, false, malicious);
    delta = masked_delta;
}

template <typename T>
void twopc_protocols::utils::gelu_same_scale(ShareTensor<T> &x, ShareTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
    constexpr uint32_t bit_width = static_cast<uint32_t>(sizeof(T) * 8);
    constexpr uint32_t kGeluEvalBitlength = bit_width > 48 ? 48 : bit_width;
    ShareTensor<T> relu_x(x.shape), abs_x(x.shape), delta(x.shape);
    {
        perf_stats::Scope stage_scope("twopc_protocols::gelu::select_relu");
        select(x, x, relu_x, parameters, malicious, "twopc_protocols::gelu::select_relu", kGeluEvalBitlength);
    }
    mulConstSubBias(relu_x, static_cast<T>(2), x, abs_x);
    twopc_protocols::utils::gelu_fourier_delta(abs_x, delta, parameters, malicious,
                                               "twopc_protocols::gelu", x.float_scale_bit,
                                               kGeluEvalBitlength);
    twopc_protocols::add(relu_x, delta, res);
}

template <typename T>
void twopc_protocols::gelu(ShareTensor<T> &x, ShareTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
    perf_stats::Scope perf_scope("twopc_protocols::gelu");
    utils::gelu_same_scale(x, res, parameters, malicious);
}

template <typename T>
void twopc_protocols::silu(ShareTensor<T> &x, ShareTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
    (void)parameters;
    (void)malicious;
    // SiLU is intentionally disabled for now to keep the network path callable without LUT dependencies.
    res = x;
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
    (void)parameter;
    ShareTensor<T> neg_x(x.shape);
    sub(static_cast<T>(0), x, neg_x);
    detail::exp_limit(neg_x, res, malicious);
}

template <typename T>
void twopc_protocols::softmax_forward(ShareTensor<T> &x, ShareTensor<T> &res, Parameters<T> &parameter, bool malicious)
{
    perf_stats::Scope perf_scope("twopc_protocols::softmax_forward");
    const auto profile_begin = std::chrono::steady_clock::now();

    constexpr uint32_t kSoftmaxIterations = 16;

    always_assert(!x.shape.empty());
    res.assert_same_shape(x);

    std::vector<uint32_t> sum_shape = x.shape;
    const uint32_t dim_size = sum_shape.back();
    always_assert(dim_size > 0);
    sum_shape.pop_back();
    const uint32_t common_size = x.size() / dim_size;

    ShareTensor<T> x_scaled(x.shape);
    {
        perf_stats::Scope stage_scope("twopc_protocols::softmax_forward::clip_and_scale");
        detail::prepare_softmax_input(x, x_scaled, parameter, malicious, "twopc_protocols::softmax_forward");
    }

    ShareTensor<T> y(x.shape), prod(x.shape), delta(x.shape), update(x.shape), dot(sum_shape), dot_broadcast(x.shape);
    y.zeros();

    const T init_prob = detail::fixed_to_ring<T>(1.0 / static_cast<double>(dim_size));
    if (Party2PC::getInstance().party_id == 0)
    {
#pragma omp parallel for if (parallel::should_parallelize(static_cast<uint64_t>(common_size) * dim_size))
        for (int i = 0; i < static_cast<int>(common_size); ++i)
        {
            for (int j = 0; j < static_cast<int>(dim_size); ++j)
            {
                y.share.data[i * dim_size + j] = init_prob;
            }
        }
    }

    for (uint32_t iter = 0; iter < kSoftmaxIterations; ++iter)
    {
        std::string iter_prefix = "twopc_protocols::softmax_forward::iter" + std::to_string(iter);

        {
            perf_stats::Scope stage_scope((iter_prefix + "::mul_xy").c_str());
            mul(x_scaled, y, prod, true, malicious);
        }

        {
            perf_stats::Scope stage_scope((iter_prefix + "::dot_reduce").c_str());
#pragma omp parallel for if (parallel::should_parallelize(static_cast<uint64_t>(common_size) * dim_size))
            for (int i = 0; i < static_cast<int>(common_size); ++i)
            {
                T acc = 0;
                for (int j = 0; j < static_cast<int>(dim_size); ++j)
                {
                    acc += prod.share.data[i * dim_size + j];
                }
                dot.share.data[i] = acc;
            }
        }

        {
            perf_stats::Scope stage_scope((iter_prefix + "::broadcast").c_str());
#pragma omp parallel for if (parallel::should_parallelize(static_cast<uint64_t>(common_size) * dim_size))
            for (int i = 0; i < static_cast<int>(common_size); ++i)
            {
                for (int j = 0; j < static_cast<int>(dim_size); ++j)
                {
                    dot_broadcast.share.data[i * dim_size + j] = dot.share.data[i];
                }
            }
        }

        {
            perf_stats::Scope stage_scope((iter_prefix + "::update").c_str());
            sub(x_scaled, dot_broadcast, delta);
            mul(delta, y, update, true, malicious);
            add(y, update, y);
        }
    }

    res = y;

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

