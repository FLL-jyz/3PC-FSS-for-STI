//
// Created by nssx on 24-12-28.
//

#ifndef LAYERS_H
#define LAYERS_H

#include <functional>
#include <chrono>
#include <atomic>
#include <omp.h>
#include <thread>
#include <unordered_map>
#include <utility>
#include <mutex>
#include <cmath>
#include <vector>

#include "llmConfig.h"
#include "perf_stats.h"
#include "twopc_protocols.h"
#include "params.h"

namespace utils
{
    constexpr uint64_t kMinOmpWorkItems = 1u << 14;

    inline bool should_parallelize(uint64_t work_items)
    {
        return work_items >= kMinOmpWorkItems;
    }

    template <typename T>
    void mask(twopc_protocols::ShareTensor<T> &attention_probs)
    {
        uint32_t common_size = attention_probs.shape[0] * attention_probs.shape[1];
        uint32_t mask_size = attention_probs.shape[2] * attention_probs.shape[3];
        for (int i = 0; i < common_size; i++)
        {
            for (int j = 0; j < attention_probs.shape[2]; j++)
            {
                for (int k = 0; k < attention_probs.shape[3]; k++)
                {
                    if (j < k)
                    {
                        attention_probs.share.data[i * mask_size + j * attention_probs.shape[3] + k] = 0;
                    }
                }
            }
        }
    }

    template <typename T>
    void masked_softmax(twopc_protocols::ShareTensor<T> &x, twopc_protocols::ShareTensor<T> &res, Parameters<T> &parameter, bool malicious = false);

    template <typename T>
    twopc_protocols::ShareTensor<T> img2col_pool(twopc_protocols::ShareTensor<T> &x, const int kernel_size, const int stride = 1, const int pad = 0);

    template <typename T>
    twopc_protocols::ShareTensor<T> img2col_conv(twopc_protocols::ShareTensor<T> &x, const int kernel_size, const int stride = 1, const int pad = 0);
}

namespace layer_parallel
{
    inline constexpr uint64_t kParallelWorkThreshold = 1ULL << 12;

    inline int concurrent_task_thread_limit(int task_count)
    {
        return twopc_protocols::parallel::concurrent_task_thread_limit(task_count);
    }

    inline bool should_parallelize(uint64_t outer_work_items, uint64_t inner_work_items = 1)
    {
        return outer_work_items * inner_work_items >= kParallelWorkThreshold;
    }

    inline bool should_parallelize_tasks(uint64_t outer_work_items, uint64_t inner_work_items, int task_count)
    {
        return task_count > 1 && omp_get_max_threads() > 1 && should_parallelize(outer_work_items, inner_work_items);
    }
}

namespace llama_profile
{
    struct TimingSnapshot
    {
        uint64_t attention_ns;
        uint64_t attention_calls;
        uint64_t swiglu_ns;
        uint64_t swiglu_calls;
        uint64_t rmsnorm_ns;
        uint64_t rmsnorm_calls;
    };

    inline std::atomic<uint64_t> attention_ns_total{0};
    inline std::atomic<uint64_t> attention_calls_total{0};
    inline std::atomic<uint64_t> swiglu_ns_total{0};
    inline std::atomic<uint64_t> swiglu_calls_total{0};
    inline std::atomic<uint64_t> rmsnorm_ns_total{0};
    inline std::atomic<uint64_t> rmsnorm_calls_total{0};

    inline void reset()
    {
        attention_ns_total.store(0, std::memory_order_relaxed);
        attention_calls_total.store(0, std::memory_order_relaxed);
        swiglu_ns_total.store(0, std::memory_order_relaxed);
        swiglu_calls_total.store(0, std::memory_order_relaxed);
        rmsnorm_ns_total.store(0, std::memory_order_relaxed);
        rmsnorm_calls_total.store(0, std::memory_order_relaxed);
    }

    inline void add_attention_ns(uint64_t ns)
    {
        attention_ns_total.fetch_add(ns, std::memory_order_relaxed);
        attention_calls_total.fetch_add(1, std::memory_order_relaxed);
    }

    inline void add_swiglu_ns(uint64_t ns)
    {
        swiglu_ns_total.fetch_add(ns, std::memory_order_relaxed);
        swiglu_calls_total.fetch_add(1, std::memory_order_relaxed);
    }

    inline void add_rmsnorm_ns(uint64_t ns)
    {
        rmsnorm_ns_total.fetch_add(ns, std::memory_order_relaxed);
        rmsnorm_calls_total.fetch_add(1, std::memory_order_relaxed);
    }

    inline TimingSnapshot snapshot()
    {
        return TimingSnapshot{
            attention_ns_total.load(std::memory_order_relaxed),
            attention_calls_total.load(std::memory_order_relaxed),
            swiglu_ns_total.load(std::memory_order_relaxed),
            swiglu_calls_total.load(std::memory_order_relaxed),
            rmsnorm_ns_total.load(std::memory_order_relaxed),
            rmsnorm_calls_total.load(std::memory_order_relaxed)};
    }
}

namespace llama_utils
{
    constexpr double kRopeBase = 10000.0;

    inline int64_t ring_to_signed(uint64_t v)
    {
        return static_cast<int64_t>(v);
    }

    inline uint64_t signed_to_ring(int64_t v)
    {
        return static_cast<uint64_t>(v);
    }

    inline int64_t mul_fixed(int64_t a, int64_t b)
    {
        constexpr uint32_t frac_bits = FLOAT_PRECISION_64;
        __int128 product = static_cast<__int128>(a) * static_cast<__int128>(b);
        return static_cast<int64_t>(product >> frac_bits);
    }

    inline int64_t encode_fixed(double x)
    {
        constexpr double scale = static_cast<double>(uint64_t(1) << FLOAT_PRECISION_64);
        const long long q = static_cast<long long>(std::llround(x * scale));
        return static_cast<int64_t>(q);
    }

    inline void apply_rope_inplace(twopc_protocols::ShareTensor<uint64_t> &x)
    {
        always_assert(x.shape.size() == 4);
        const uint32_t batch = x.shape[0];
        const uint32_t heads = x.shape[1];
        const uint32_t seq_len = x.shape[2];
        const uint32_t head_dim = x.shape[3];
        always_assert(head_dim % 2 == 0);
        const uint32_t half_dim = head_dim / 2;

        std::vector<int64_t> cos_table(seq_len * half_dim);
        std::vector<int64_t> sin_table(seq_len * half_dim);
        for (uint32_t pos = 0; pos < seq_len; ++pos)
        {
            for (uint32_t i = 0; i < half_dim; ++i)
            {
                const double theta = std::pow(kRopeBase, -2.0 * static_cast<double>(i) / static_cast<double>(head_dim));
                const double angle = static_cast<double>(pos) * theta;
                cos_table[pos * half_dim + i] = encode_fixed(std::cos(angle));
                sin_table[pos * half_dim + i] = encode_fixed(std::sin(angle));
            }
        }

        for (uint32_t b = 0; b < batch; ++b)
        {
            for (uint32_t h = 0; h < heads; ++h)
            {
                for (uint32_t pos = 0; pos < seq_len; ++pos)
                {
                    for (uint32_t i = 0; i < half_dim; ++i)
                    {
                        const uint32_t even_idx = ((b * heads + h) * seq_len + pos) * head_dim + 2 * i;
                        const uint32_t odd_idx = even_idx + 1;

                        const int64_t x_even = ring_to_signed(x.share.data[even_idx]);
                        const int64_t x_odd = ring_to_signed(x.share.data[odd_idx]);
                        const int64_t cos_v = cos_table[pos * half_dim + i];
                        const int64_t sin_v = sin_table[pos * half_dim + i];

                        const int64_t rot_even = mul_fixed(x_even, cos_v) - mul_fixed(x_odd, sin_v);
                        const int64_t rot_odd = mul_fixed(x_even, sin_v) + mul_fixed(x_odd, cos_v);

                        x.share.data[even_idx] = signed_to_ring(rot_even);
                        x.share.data[odd_idx] = signed_to_ring(rot_odd);
                    }
                }
            }
        }
    }

    inline void expand_kv_heads(const twopc_protocols::ShareTensor<uint64_t> &src, twopc_protocols::ShareTensor<uint64_t> &dst)
    {
        always_assert(src.shape.size() == 4);
        always_assert(dst.shape.size() == 4);
        always_assert(src.shape[0] == dst.shape[0]);
        always_assert(src.shape[2] == dst.shape[2]);
        always_assert(src.shape[3] == dst.shape[3]);
        always_assert(dst.shape[1] % src.shape[1] == 0);

        const uint32_t batch = src.shape[0];
        const uint32_t kv_heads = src.shape[1];
        const uint32_t q_heads = dst.shape[1];
        const uint32_t dim2 = src.shape[2];
        const uint32_t dim3 = src.shape[3];
        const uint32_t group_size = q_heads / kv_heads;

        for (uint32_t b = 0; b < batch; ++b)
        {
            for (uint32_t kv = 0; kv < kv_heads; ++kv)
            {
                for (uint32_t g = 0; g < group_size; ++g)
                {
                    const uint32_t qh = kv * group_size + g;
                    for (uint32_t i = 0; i < dim2; ++i)
                    {
                        for (uint32_t j = 0; j < dim3; ++j)
                        {
                            const uint32_t src_idx = ((b * kv_heads + kv) * dim2 + i) * dim3 + j;
                            const uint32_t dst_idx = ((b * q_heads + qh) * dim2 + i) * dim3 + j;
                            dst.share.data[dst_idx] = src.share.data[src_idx];
                        }
                    }
                }
            }
        }
    }
}

template <typename T, typename U = T, typename V = U>
class Layers
{
private:
    std::mutex param_mutex;

public:
    std::string name;
    std::unordered_map<std::string, twopc_protocols::ShareTensor<T>> parameters;
    bool need_parameters = false;

    Layers() = default;

    Layers(std::string name) : name(std::move(name))
    {
    }

    Layers(const Layers<T, U, V> &other) : name(other.name), parameters(other.parameters), need_parameters(other.need_parameters) {}

    virtual ~Layers() = default;

    virtual twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x) { return x; }

    virtual twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<U> &parameters) { return x; }

    virtual twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<U> &parameters_u, Parameters<V> &parameters_v) { return x; }

    virtual twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters_t, Parameters<U> &parameters_u, Parameters<V> &parameters_v) { return x; }

    virtual void init_param() { return; }

    void load_param()
    {
        if (!need_parameters)
        {
            return;
        }

        std::lock_guard<std::mutex> lock(this->param_mutex);
        if (!need_parameters)
        {
            return;
        }

        Loader<T> &loader = Party2PC::getInstance().get_loader<T>();
        if (!loader.is_buffer_empty)
        {
            auto loaded = loader.get_data();
            this->parameters.clear();
            for (auto &it : loaded)
            {
                twopc_protocols::ShareTensor<T> param_tensor(it.second.shape);
                param_tensor.share = it.second;
                this->parameters[it.first] = param_tensor;
            }
        }
        need_parameters = false;
        init_param();
    }

    twopc_protocols::ShareTensor<T> operator()(twopc_protocols::ShareTensor<T> &x)
    {
        load_param();
        return forward(x);
    }

    twopc_protocols::ShareTensor<T> operator()(twopc_protocols::ShareTensor<T> &x, Parameters<U> &parameters)
    {
        load_param();
        return forward(x, parameters);
    }

    twopc_protocols::ShareTensor<T> operator()(twopc_protocols::ShareTensor<T> &x, Parameters<U> &parameters_u, Parameters<V> &parameters_v)
    {
        load_param();
        return forward(x, parameters_u, parameters_v);
    }

    twopc_protocols::ShareTensor<T> operator()(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters_t, Parameters<U> &parameters_u, Parameters<V> &parameters_v)
    {
        load_param();
        return forward(x, parameters_t, parameters_u, parameters_v);
    }
};

template <typename T>
class SecConv : public Layers<T>
{
public:
    uint32_t in_channels;
    uint32_t out_channels;
    uint32_t kernel_size;
    uint32_t stride;
    uint32_t padding;

    twopc_protocols::ShareTensor<T> weight;
    twopc_protocols::ShareTensor<T> bias;

    SecConv(uint32_t in_channels, uint32_t out_channels, uint32_t kernel_size) : Layers<T>("Conv"),
                                                                                 in_channels(in_channels),
                                                                                 out_channels(out_channels), kernel_size(kernel_size)
    {
        this->stride = 1;
        this->padding = 0;

        weight.allocate({out_channels, in_channels, kernel_size, kernel_size});
        bias.allocate({out_channels});
        weight.zeros();
        bias.zeros();

        this->parameters["weight"] = weight;
        this->parameters["bias"] = bias;

        this->need_parameters = true;
    }

    SecConv(uint32_t in_channels, uint32_t out_channels, uint32_t kernel_size, uint32_t stride, uint32_t padding) : Layers<T>("Conv"),
                                                                                                                    in_channels(in_channels),
                                                                                                                    out_channels(out_channels),
                                                                                                                    kernel_size(kernel_size),
                                                                                                                    stride(stride), padding(padding)
    {
        weight.allocate({out_channels, in_channels, kernel_size, kernel_size});
        bias.allocate({out_channels});
        weight.zeros();
        bias.zeros();

        this->parameters["weight"] = weight;
        this->parameters["bias"] = bias;

        this->need_parameters = true;
    }

    SecConv(twopc_protocols::ShareTensor<T> &weight, twopc_protocols::ShareTensor<T> &bias) : Layers<T>("Conv"), weight(weight), bias(bias)
    {
        in_channels = weight.shape[1];
        out_channels = weight.shape[0];
        kernel_size = weight.shape[2];
        stride = 1;
        padding = 0;

        this->parameters["weight"] = weight;
        this->parameters["bias"] = bias;
    }

    void set_weight(twopc_protocols::ShareTensor<T> &weight)
    {
        always_assert(this->in_channels == weight.shape[1]);
        always_assert(this->out_channels == weight.shape[0]);
        always_assert(this->kernel_size == weight.shape[2]);
        this->weight = weight;
        this->parameters["weight"] = weight;
    }

    void set_bias(twopc_protocols::ShareTensor<T> &bias)
    {
        always_assert(this->out_channels == bias.shape[0]);
        this->bias = bias;
        this->parameters["bias"] = bias;
    }

    ~SecConv() override = default;

    void init_param()
    {
        this->parameters["weight"].reshape({this->out_channels, this->in_channels * this->kernel_size * this->kernel_size});
        weight = this->parameters["weight"].transpose();
        bias = this->parameters["bias"];
    }

    std::vector<uint32_t> get_out_shape(std::vector<uint32_t> in_shape)
    {
        uint32_t batch = in_shape[0], in_height = in_shape[2], in_width = in_shape[3];
        uint32_t out_height = (in_height - this->kernel_size + 2 * this->padding) / this->stride + 1;
        uint32_t out_width = (in_width - this->kernel_size + 2 * this->padding) / this->stride + 1;

        return {batch, this->out_channels, out_height, out_width};
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x) override
    {
        std::vector<uint32_t> out_shape = get_out_shape(x.shape);

        twopc_protocols::ShareTensor<T> x_col = utils::img2col_conv(x, this->kernel_size, this->stride, this->padding);
        twopc_protocols::ShareTensor<T> output({x_col.shape[0], x_col.shape[1], this->out_channels});

        twopc_protocols::matMul(x_col, weight, output, true, IS_MALICIOUS);
        uint32_t broadcast_size = x_col.shape[0] * x_col.shape[1];

        for (int i = 0; i < broadcast_size; i++)
        {
            for (int j = 0; j < this->out_channels; j++)
            {
                output.share.data[i * this->out_channels + j] += bias.share.data[j];
            }
        }
        output = output.transpose();
        output.reshape(out_shape);
        return output;
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters) override
    {
        return forward(x);
    }
};

template <typename T>
class SecMaxPool : public Layers<T>
{
public:
    uint32_t kernel_size;
    uint32_t stride;
    uint32_t padding;

    SecMaxPool(uint32_t kernel_size, uint32_t stride = 1, uint32_t padding = 0) : Layers<T>("MaxPool"), kernel_size(kernel_size), stride(stride), padding(padding)
    {
    }

    std::vector<uint32_t> get_out_shape(std::vector<uint32_t> in_shape)
    {
        uint32_t batch = in_shape[0], in_channel = in_shape[1], in_height = in_shape[2], in_width = in_shape[3];
        uint32_t out_height = (in_height - this->kernel_size + 2 * this->padding) / this->stride + 1;
        uint32_t out_width = (in_width - this->kernel_size + 2 * this->padding) / this->stride + 1;

        return {batch, in_channel, out_height, out_width};
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters) override
    {
        std::vector<uint32_t> out_shape = get_out_shape(x.shape);
        twopc_protocols::ShareTensor<T> x_col = utils::img2col_pool(x, this->kernel_size, this->stride, this->padding);
        std::vector<uint32_t> col_max_shape = x_col.shape;
        col_max_shape.pop_back();

        twopc_protocols::ShareTensor<T> res(col_max_shape);

        twopc_protocols::max_last_dim(x_col, res, parameters, IS_MALICIOUS);
        res.reshape(out_shape);
        return res;
    }
};

template <typename T>
class SecAvgPool : public Layers<T>
{
public:
    uint32_t kernel_size;
    uint32_t stride;
    uint32_t padding;

    SecAvgPool(uint32_t kernel_size, uint32_t stride = 1, uint32_t padding = 0) : Layers<T>("AvgPool"), kernel_size(kernel_size), stride(stride), padding(padding)
    {
    }

    std::vector<uint32_t> get_out_shape(std::vector<uint32_t> in_shape)
    {
        uint32_t batch = in_shape[0], in_channel = in_shape[1], in_height = in_shape[2], in_width = in_shape[3];
        uint32_t out_height = (in_height - this->kernel_size + 2 * this->padding) / this->stride + 1;
        uint32_t out_width = (in_width - this->kernel_size + 2 * this->padding) / this->stride + 1;

        return {batch, in_channel, out_height, out_width};
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x) override
    {
        std::vector<uint32_t> out_shape = get_out_shape(x.shape);
        twopc_protocols::ShareTensor<T> x_col = utils::img2col_pool(x, this->kernel_size, this->stride, this->padding);
        std::vector<uint32_t> res_shape = x_col.shape;
        uint32_t kernel_shape = res_shape.back();
        res_shape.pop_back();

        twopc_protocols::ShareTensor<T> res(res_shape);
        uint32_t res_size = res.size();
        for (int i = 0; i < res_size; i++)
        {
            res.share.data[i] = 0;
            for (int j = 0; j < kernel_shape; j++)
            {
                res.share.data[i] += x_col.share.data[i * kernel_shape + j];
            }
        }
        twopc_protocols::truncate(res, kernel_shape, IS_MALICIOUS);

        return res;
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters) override
    {
        return forward(x);
    }
};

template <typename T>
class SecAdaptiveAvgPool : public Layers<T>
{
public:
    uint32_t out_size;

    SecAdaptiveAvgPool(uint32_t out_size) : Layers<T>("AdaptiveAvgPool"), out_size(out_size)
    {
    }

    std::vector<uint32_t> get_out_shape(std::vector<uint32_t> in_shape, uint32_t kernel_size, uint32_t stride, uint32_t padding)
    {
        uint32_t batch = in_shape[0], in_channel = in_shape[1], in_height = in_shape[2], in_width = in_shape[3];
        uint32_t out_height = (in_height - kernel_size + 2 * padding) / stride + 1;
        uint32_t out_width = (in_width - kernel_size + 2 * padding) / stride + 1;

        return {batch, in_channel, out_height, out_width};
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x) override
    {
        uint32_t in_size = x.shape.back();
        uint32_t stride = floor((float)(in_size) / (float)(this->out_size));
        uint32_t kernel_size = in_size - (this->out_size - 1) * stride;

        std::vector<uint32_t> out_shape = get_out_shape(x.shape, kernel_size, stride, 0);
        twopc_protocols::ShareTensor<T> x_col = utils::img2col_pool(x, kernel_size, stride, 0);
        std::vector<uint32_t> res_shape = x_col.shape;
        uint32_t kernel_shape = res_shape.back();
        res_shape.pop_back();

        twopc_protocols::ShareTensor<T> res(res_shape);
        uint32_t res_size = res.size();
        for (int i = 0; i < res_size; i++)
        {
            res.share.data[i] = 0;
            for (int j = 0; j < kernel_shape; j++)
            {
                res.share.data[i] += x_col.share.data[i * kernel_shape + j];
            }
        }
        twopc_protocols::truncate(res, kernel_shape, IS_MALICIOUS);

        return res;
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters) override
    {
        return forward(x);
    }
};

template <typename T>
class SecBatchNorm : public Layers<T>
{
public:
    uint32_t num_features;
    twopc_protocols::ShareTensor<T> weight;
    twopc_protocols::ShareTensor<T> bias;

    SecBatchNorm(uint32_t num_features) : Layers<T>("BatchNorm"), num_features(num_features)
    {
        weight.allocate({num_features});
        bias.allocate({num_features});
        weight.zeros();
        bias.zeros();
        this->parameters["weight"] = weight;
        this->parameters["bias"] = bias;

        this->need_parameters = true;
    }

    void set_weight(twopc_protocols::ShareTensor<T> &weight)
    {
        always_assert(this->num_features == weight.shape[0]);
        this->weight = weight;
        this->parameters["weight"] = weight;
    }

    void set_bias(twopc_protocols::ShareTensor<T> &bias)
    {
        always_assert(this->num_features == bias.shape[0]);
        this->bias = bias;
        this->parameters["bias"] = bias;
    }

    SecBatchNorm(twopc_protocols::ShareTensor<T> &weight, twopc_protocols::ShareTensor<T> &bias) : Layers<T>("BatchNorm")
    {
        num_features = weight.shape[0];
        weight.allocate({num_features});
        bias.allocate({num_features});
        this->weight = weight;
        this->bias = bias;
        this->parameters["weight"] = weight;
        this->parameters["bias"] = bias;
    }

    void init_param()
    {
        weight = this->parameters["weight"];
        bias = this->parameters["bias"];
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x) override
    {
        twopc_protocols::ShareTensor<T> res(x.shape);

        uint32_t batch = x.shape[0], channel = x.shape[1], each_shape = x.shape[2] * x.shape[3];
        always_assert(channel == this->num_features);
        twopc_protocols::ShareTensor<T> gamma(x.shape);

        int index;
        for (int i = 0; i < batch; i++)
        {
            for (int j = 0; j < channel; j++)
            {
                for (int k = 0; k < each_shape; k++)
                {
                    index = i * channel * each_shape + j * each_shape + k;
                    gamma.share.data[index] = this->weight.share.data[j];
                }
            }
        }

        twopc_protocols::ShareTensor<T> x_weighted(x.shape);
        twopc_protocols::mul(x, gamma, x_weighted, true, IS_MALICIOUS); // calculate gamma * x

        // calculate gamma * x + beta
        for (int i = 0; i < batch; i++)
        {
            for (int j = 0; j < channel; j++)
            {
                for (int k = 0; k < each_shape; k++)
                {
                    index = i * channel * each_shape + j * each_shape + k;
                    res.share.data[index] = x_weighted.share.data[index] + this->bias.share.data[j];
                }
            }
        }

        return res;
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters) override
    {
        return forward(x);
    }
};

template <typename T>
class SecLinear : public Layers<T>
{
public:
    uint32_t in_features;
    uint32_t out_features;
    twopc_protocols::ShareTensor<T> weight;
    twopc_protocols::ShareTensor<T> bias;

    SecLinear(uint32_t in_features, uint32_t out_features) : Layers<T>("Linear"),
                                                             in_features(in_features),
                                                             out_features(out_features)
    {
        weight.allocate({out_features, in_features});
        bias.allocate({out_features});
        weight.zeros();
        bias.zeros();
        this->parameters["weight"] = weight;
        this->parameters["bias"] = bias;

        this->need_parameters = true;
    }

    SecLinear(twopc_protocols::ShareTensor<T> &weight, twopc_protocols::ShareTensor<T> &bias) : Layers<T>("Linear")
    {
        in_features = weight.shape[1];
        out_features = weight.shape[0];
        weight.allocate({out_features, in_features});
        bias.allocate({out_features});
        this->weight = weight;
        this->bias = bias;
        this->parameters["weight"] = weight;
        this->parameters["bias"] = bias;
    }

    void set_weight(twopc_protocols::ShareTensor<T> &weight)
    {
        always_assert(this->in_features == weight.shape[1]);
        always_assert(this->out_features == weight.shape[0]);
        this->weight = weight;
        this->parameters["weight"] = weight;
    }

    void set_bias(twopc_protocols::ShareTensor<T> &bias)
    {
        always_assert(this->out_features == bias.shape[0]);
        this->bias = bias;
        this->parameters["bias"] = bias;
    }

    void init_param()
    {
        weight = this->parameters["weight"].transpose();
        bias = this->parameters["bias"];
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x) override
    {
        std::vector<uint32_t> res_shape;
        uint32_t broadcast_size = 1;
        for (int i = 0; i < x.shape.size() - 1; i++)
        {
            res_shape.push_back(x.shape[i]);
            broadcast_size *= x.shape[i];
        }
        res_shape.push_back(this->weight.shape[1]);
        twopc_protocols::ShareTensor<T> res(res_shape);
        twopc_protocols::matMul(x, weight, res, true, IS_MALICIOUS);

        for (int i = 0; i < broadcast_size; i++)
        {
            for (int j = 0; j < this->out_features; j++)
            {
                res.share.data[i * this->out_features + j] += this->bias.share.data[j];
            }
        }
        return res;
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters) override
    {
        return forward(x);
    }
};

template <typename T>
class SecReLU : public Layers<T>
{
public:
    SecReLU() : Layers<T>("ReLU")
    {
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters) override
    {
        twopc_protocols::ShareTensor<T> res(x.shape);
        twopc_protocols::select(x, x, res, parameters, IS_MALICIOUS);
        return res;
    }
};

template <typename T>
void fused_linear_pair_forward(twopc_protocols::ShareTensor<T> &x,
                               SecLinear<T> &first_linear,
                               SecLinear<T> &second_linear,
                               twopc_protocols::ShareTensor<T> &first_out,
                               twopc_protocols::ShareTensor<T> &second_out)
{
    const uint32_t in_features = first_linear.in_features;
    const uint32_t first_out_features = first_linear.out_features;
    const uint32_t second_out_features = second_linear.out_features;
    const uint32_t fused_out_features = first_out_features + second_out_features;
    always_assert(second_linear.in_features == in_features);
    always_assert(x.shape.back() == in_features);

    twopc_protocols::ShareTensor<T> fused_weight({in_features, fused_out_features});
    twopc_protocols::ShareTensor<T> fused_bias({fused_out_features});

#pragma omp parallel for if (layer_parallel::should_parallelize(static_cast<uint64_t>(in_features) * fused_out_features))
    for (int i = 0; i < static_cast<int>(in_features); i++)
    {
        const uint32_t fused_row = static_cast<uint32_t>(i) * fused_out_features;
        const uint32_t first_row = static_cast<uint32_t>(i) * first_out_features;
        const uint32_t second_row = static_cast<uint32_t>(i) * second_out_features;
        for (uint32_t j = 0; j < first_out_features; j++)
        {
            fused_weight.share.data[fused_row + j] = first_linear.weight.share.data[first_row + j];
        }
        for (uint32_t j = 0; j < second_out_features; j++)
        {
            fused_weight.share.data[fused_row + first_out_features + j] = second_linear.weight.share.data[second_row + j];
        }
    }

#pragma omp parallel for if (layer_parallel::should_parallelize(fused_out_features))
    for (int j = 0; j < static_cast<int>(first_out_features); j++)
    {
        fused_bias.share.data[j] = first_linear.bias.share.data[j];
    }
#pragma omp parallel for if (layer_parallel::should_parallelize(fused_out_features))
    for (int j = 0; j < static_cast<int>(second_out_features); j++)
    {
        fused_bias.share.data[first_out_features + static_cast<uint32_t>(j)] = second_linear.bias.share.data[j];
    }

    std::vector<uint32_t> fused_shape = x.shape;
    fused_shape.back() = fused_out_features;
    twopc_protocols::ShareTensor<T> fused_out(fused_shape);
    twopc_protocols::matMul(x, fused_weight, fused_out, true, IS_MALICIOUS);

    const uint32_t broadcast_size = fused_out.size() / fused_out_features;
#pragma omp parallel for if (layer_parallel::should_parallelize(static_cast<uint64_t>(broadcast_size) * fused_out_features))
    for (int i = 0; i < static_cast<int>(broadcast_size); i++)
    {
        const uint32_t row = static_cast<uint32_t>(i) * fused_out_features;
        for (uint32_t j = 0; j < fused_out_features; j++)
        {
            fused_out.share.data[row + j] += fused_bias.share.data[j];
        }
    }

    std::vector<uint32_t> first_shape = x.shape;
    first_shape.back() = first_out_features;
    std::vector<uint32_t> second_shape = x.shape;
    second_shape.back() = second_out_features;
    first_out.allocate(first_shape);
    second_out.allocate(second_shape);

#pragma omp parallel for if (layer_parallel::should_parallelize(static_cast<uint64_t>(broadcast_size) * fused_out_features))
    for (int i = 0; i < static_cast<int>(broadcast_size); i++)
    {
        const uint32_t fused_row = static_cast<uint32_t>(i) * fused_out_features;
        const uint32_t first_row = static_cast<uint32_t>(i) * first_out_features;
        const uint32_t second_row = static_cast<uint32_t>(i) * second_out_features;
        for (uint32_t j = 0; j < first_out_features; j++)
        {
            first_out.share.data[first_row + j] = fused_out.share.data[fused_row + j];
        }
        for (uint32_t j = 0; j < second_out_features; j++)
        {
            second_out.share.data[second_row + j] = fused_out.share.data[fused_row + first_out_features + j];
        }
    }
}

template <typename T>
class SecGELU : public Layers<T>
{
public:
    SecGELU() : Layers<T>("GELU")
    {
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters) override
    {
        twopc_protocols::ShareTensor<T> res(x.shape);
        twopc_protocols::gelu(x, res, parameters, IS_MALICIOUS);
        return res;
    }
};

template <typename T>
class SecSiLU : public Layers<T>
{
public:
    SecSiLU() : Layers<T>("SiLU")
    {
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters) override
    {
        twopc_protocols::ShareTensor<T> res(x.shape);
        twopc_protocols::silu(x, res, parameters, IS_MALICIOUS);
        return res;
    }
};

template <typename T>
class SecSoftmax : public Layers<T>
{
public:
    bool masked;

    SecSoftmax(bool masked = false) : Layers<T>("Softmax"), masked(masked)
    {
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters) override
    {
        twopc_protocols::ShareTensor<T> res(x.shape);
        if (masked)
            utils::masked_softmax(x, res, parameters, IS_MALICIOUS);
        else
            twopc_protocols::softmax_forward(x, res, parameters, IS_MALICIOUS);
        return res;
    }
};

template <typename T>
class SecLayerNorm : public Layers<T>
{
public:
    uint32_t normalized_shape;
    twopc_protocols::ShareTensor<T> weight;
    twopc_protocols::ShareTensor<T> bias;

    SecLayerNorm(uint32_t normalized_shape) : Layers<T>("LayerNorm"),
                                              normalized_shape(normalized_shape)
    {
        weight.allocate({normalized_shape});
        bias.allocate({normalized_shape});
        weight.zeros();
        bias.zeros();
        this->parameters["weight"] = weight;
        this->parameters["bias"] = bias;

        this->need_parameters = true;
    }

    void set_weight(twopc_protocols::ShareTensor<T> &weight)
    {
        always_assert(this->normalized_shape == weight.shape[0]);
        this->weight = weight;
        this->parameters["weight"] = weight;
    }

    void set_bias(twopc_protocols::ShareTensor<T> &bias)
    {
        always_assert(this->normalized_shape == bias.shape[0]);
        this->bias = bias;
        this->parameters["bias"] = bias;
    }

    SecLayerNorm(twopc_protocols::ShareTensor<T> &weight, twopc_protocols::ShareTensor<T> &bias) : Layers<T>(
                                                                 "LayerNorm")
    {
        normalized_shape = weight.shape[0];
        weight.allocate({normalized_shape});
        bias.allocate({normalized_shape});
        this->weight = weight;
        this->bias = bias;
        this->parameters["weight"] = weight;
        this->parameters["bias"] = bias;
    }

    void init_param()
    {
        weight = this->parameters["weight"];
        bias = this->parameters["bias"];
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters) override
    {
        twopc_protocols::ShareTensor<T> res(x.shape);

        std::vector<uint32_t> sum_shape = x.shape;
        uint32_t dim_size = sum_shape.back();
        always_assert(dim_size == this->normalized_shape); // can only normalize the last dimension
        sum_shape.pop_back();

        twopc_protocols::ShareTensor<T> x_sum(sum_shape);
        const uint32_t sum_size = x_sum.size();
    #pragma omp parallel for if (utils::should_parallelize(static_cast<uint64_t>(sum_size) * dim_size))
        for (int i = 0; i < sum_size; i++)
        {
            x_sum.share.data[i] = 0;
            for (int j = 0; j < dim_size; j++)
            {
                x_sum.share.data[i] += x.share.data[i * dim_size + j];
            }
        }

        twopc_protocols::truncate(x_sum, normalized_shape, IS_MALICIOUS); // calculate mean

        twopc_protocols::ShareTensor<T> z(x.shape);
#pragma omp parallel for if (utils::should_parallelize(static_cast<uint64_t>(sum_size) * dim_size))
        for (int i = 0; i < sum_size; i++)
        {
            for (int j = 0; j < dim_size; j++)
            {
                z.share.data[i * dim_size + j] = x.share.data[i * dim_size + j] - x_sum.share.data[i];
            }
        }

        twopc_protocols::ShareTensor<T> z_square(x.shape), square_sum(sum_shape);
        twopc_protocols::square(z, z_square, true, IS_MALICIOUS); // calculate z^2

#pragma omp parallel for if (utils::should_parallelize(static_cast<uint64_t>(sum_size) * dim_size))
        for (int i = 0; i < sum_size; i++)
        {
            square_sum.share.data[i] = 0;
            for (int j = 0; j < dim_size; j++)
            {
                square_sum.share.data[i] += z_square.share.data[i * dim_size + j];
            }
        }

        twopc_protocols::truncate(square_sum, normalized_shape, IS_MALICIOUS); // calculate variance

        twopc_protocols::ShareTensor<T> rsqrt_var(sum_shape), broadcast_rsqrt_var(x.shape), gamma(x.shape);
        twopc_protocols::rsqrt(square_sum, rsqrt_var, parameters, IS_MALICIOUS); // calculate 1 / sqrt(var)

    #pragma omp parallel for if (utils::should_parallelize(static_cast<uint64_t>(sum_size) * dim_size))
        for (int i = 0; i < sum_size; i++)
        {
            for (int j = 0; j < dim_size; j++)
            {
                broadcast_rsqrt_var.share.data[i * dim_size + j] = rsqrt_var.share.data[i];

                gamma.share.data[i * dim_size + j] = this->weight.share.data[j];
            }
        }

        twopc_protocols::ShareTensor<T> z_norm(x.shape), z_norm_weighted(x.shape);
        twopc_protocols::mul(z, broadcast_rsqrt_var, z_norm, true, IS_MALICIOUS); // calculate z_norm
        twopc_protocols::mul(z_norm, gamma, z_norm_weighted, true, IS_MALICIOUS); // calculate gamma * z_norm

        // calculate gamma * z_norm + beta
    #pragma omp parallel for if (utils::should_parallelize(static_cast<uint64_t>(sum_size) * dim_size))
        for (int i = 0; i < sum_size; i++)
        {
            for (int j = 0; j < dim_size; j++)
            {
                res.share.data[i * dim_size + j] = z_norm_weighted.share.data[i * dim_size + j] + this->bias.share.data[j];
            }
        }

        return res;
    }
};

template <typename T>
class SecSelfAttention : public Layers<T>
{
public:
    uint32_t num_heads;
    uint32_t head_size;
    uint32_t all_head_size;
    Layers<T> *query, *key, *value, *softmax;

    SecSelfAttention(TransformerConfig &config, const bool masked = false) : Layers<T>("SelfAttention"),
                                                                             num_heads(config.num_attention_heads),
                                                                             head_size(
                                                                                 config.hidden_size / config.num_attention_heads),
                                                                             all_head_size(num_heads * head_size)
    {
        query = new SecLinear<T>(config.hidden_size, all_head_size);
        key = new SecLinear<T>(config.hidden_size, all_head_size);
        value = new SecLinear<T>(config.hidden_size, all_head_size);
        softmax = new SecSoftmax<T>(masked);
    }

    SecSelfAttention(TransformerConfig &config, std::vector<Layers<T>> layers, const bool masked = false) : Layers<T>("SelfAttention"),
                                                                                                            num_heads(config.num_attention_heads),
                                                                                                            head_size(config.hidden_size / config.num_attention_heads),
                                                                                                            all_head_size(num_heads * head_size)
    {
        query = &layers[0];
        key = &layers[1];
        value = &layers[2];
        softmax = &layers[3];
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters) override
    {
        twopc_protocols::ShareTensor<T> mixed_query = (*this->query)(x);
        twopc_protocols::ShareTensor<T> mixed_key = (*this->key)(x);
        twopc_protocols::ShareTensor<T> mixed_value = (*this->value)(x);
        std::vector<uint32_t> new_shape = x.shape;
        new_shape.pop_back();
        new_shape.push_back(this->num_heads);
        new_shape.push_back(this->head_size);

        mixed_query.reshape(new_shape);
        mixed_key.reshape(new_shape);
        mixed_value.reshape(new_shape);

        twopc_protocols::ShareTensor<T> query_layer = mixed_query.permute({0, 2, 1, 3});
        twopc_protocols::ShareTensor<T> key_layer = mixed_key.permute({0, 2, 3, 1});
        twopc_protocols::ShareTensor<T> value_layer = mixed_value.permute({0, 2, 1, 3});

        twopc_protocols::ShareTensor<T> attention_scores(
            {query_layer.shape[0], query_layer.shape[1], query_layer.shape[2], key_layer.shape[3]});
        twopc_protocols::matMul(query_layer, key_layer, attention_scores, true, IS_MALICIOUS);
        twopc_protocols::truncate(attention_scores, (uint32_t)sqrt((double)this->head_size), IS_MALICIOUS);

        twopc_protocols::ShareTensor<T> attention_probs = (*this->softmax)(attention_scores, parameters);

        twopc_protocols::ShareTensor<T> context_layer({attention_probs.shape[0], attention_probs.shape[1], attention_probs.shape[2],
                                    value_layer.shape[3]});
        twopc_protocols::matMul(attention_probs, value_layer, context_layer, true, IS_MALICIOUS);

        twopc_protocols::ShareTensor<T> res = context_layer.permute({0, 2, 1, 3});
        std::vector<uint32_t> res_shape = res.shape;
        res_shape[res_shape.size() - 2] = this->all_head_size;
        res_shape.pop_back();
        res.reshape(res_shape);

        return res;
    }
};

template <typename T>
class SecAttention : public Layers<T>
{
public:
    Layers<T> *self, *dense;

    SecAttention(TransformerConfig &config, const bool masked = false) : Layers<T>(
                                                                             "Attention")
    {
        self = new SecSelfAttention<T>(config, masked);
        dense = new SecLinear<T>(config.hidden_size, config.hidden_size);
    }

    SecAttention(TransformerConfig &config, std::vector<Layers<T>> layers) : Layers<T>(
                                                                                 "Attention")
    {
        self = &layers[0];
        dense = &layers[1];
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters) override
    {
        twopc_protocols::ShareTensor<T> self_outputs = (*this->self)(x, parameters);
        return (*this->dense)(self_outputs, parameters);
    }
};

template <typename T>
class SecGPT2Attention : public Layers<T>
{
public:
    uint32_t num_heads;
    uint32_t head_size;
    uint32_t all_head_size;
    Layers<T> *c_attn, *softmax, *c_proj;

    SecGPT2Attention(TransformerConfig &config) : Layers<T>("GPT2Attention"),
                                                  num_heads(config.num_attention_heads),
                                                  head_size(
                                                      config.hidden_size / config.num_attention_heads),
                                                  all_head_size(num_heads * head_size)
    {
        c_attn = new SecLinear<T>(config.hidden_size, 3 * all_head_size);
        softmax = new SecSoftmax<T>(true);
        c_proj = new SecLinear<T>(config.hidden_size, all_head_size);
    }

    SecGPT2Attention(TransformerConfig &config, std::vector<Layers<T>> layers) : Layers<T>("GPT2Attention"),
                                                                                 num_heads(config.num_attention_heads),
                                                                                 head_size(config.hidden_size / config.num_attention_heads),
                                                                                 all_head_size(num_heads * head_size)
    {
        c_attn = &layers[0];
        softmax = &layers[1];
        c_proj = &layers[2];
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters) override
    {
        twopc_protocols::ShareTensor<T> qkv = (*this->c_attn)(x);
        twopc_protocols::ShareTensor<T> mixed_query(x.shape), mixed_key(x.shape), mixed_value(x.shape);

        uint32_t size = x.size(), last_dim = x.shape[x.shape.size() - 1];
        uint32_t other_dim = size / last_dim;
        // split, dim = -1
        uint32_t qkv_last_dim = qkv.shape[qkv.shape.size() - 1];
        uint32_t index, qkv_index;
        for (int i = 0; i < other_dim; i++)
        {
            for (int j = 0; j < last_dim; j++)
            {
                index = i * last_dim + j;
                qkv_index = i * qkv_last_dim + j;

                mixed_query.share.data[index] = qkv.share.data[qkv_index];

                mixed_key.share.data[index] = qkv.share.data[qkv_index + last_dim];

                mixed_value.share.data[index] = qkv.share.data[qkv_index + last_dim * 2];
            }
        }

        std::vector<uint32_t> new_shape = x.shape;
        new_shape.pop_back();
        new_shape.push_back(this->num_heads);
        new_shape.push_back(this->head_size);

        mixed_query.reshape(new_shape);
        mixed_key.reshape(new_shape);
        mixed_value.reshape(new_shape);

        twopc_protocols::ShareTensor<T> query_layer = mixed_query.permute({0, 2, 1, 3});
        twopc_protocols::ShareTensor<T> key_layer = mixed_key.permute({0, 2, 3, 1});
        twopc_protocols::ShareTensor<T> value_layer = mixed_value.permute({0, 2, 1, 3});

        twopc_protocols::ShareTensor<T> attention_scores(
            {query_layer.shape[0], query_layer.shape[1], query_layer.shape[2], key_layer.shape[3]});
        twopc_protocols::matMul(query_layer, key_layer, attention_scores, true, IS_MALICIOUS);
        twopc_protocols::truncate(attention_scores, (uint32_t)sqrt((double)this->head_size), IS_MALICIOUS);

        twopc_protocols::ShareTensor<T> attention_probs = (*this->softmax)(attention_scores, parameters);

        twopc_protocols::ShareTensor<T> context_layer({attention_probs.shape[0], attention_probs.shape[1], attention_probs.shape[2],
                                    value_layer.shape[3]});
        twopc_protocols::matMul(attention_probs, value_layer, context_layer, true, IS_MALICIOUS);

        twopc_protocols::ShareTensor<T> attn_output = context_layer.permute({0, 2, 1, 3});
        std::vector<uint32_t> output_shape = attn_output.shape;
        output_shape[output_shape.size() - 2] = this->all_head_size;
        output_shape.pop_back();
        attn_output.reshape(output_shape);

        return (*this->c_proj)(attn_output);
    }
};

template <typename T>
class SecFFN : public Layers<T>
{
public:
    Layers<T> *dense0, *act_fn, *dense1;

    SecFFN(TransformerConfig &config) : Layers<T>("FFN")
    {
        dense0 = new SecLinear<T>(config.hidden_size, config.intermediate_size);
        if (config.isReLUAct)
        {
            act_fn = new SecReLU<T>();
        }
        else
        {
            act_fn = new SecGELU<T>();
        }
        dense1 = new SecLinear<T>(config.intermediate_size, config.hidden_size);
    }

    SecFFN(TransformerConfig &config,
           std::vector<Layers<T>> layers) : Layers<T>("FFN")
    {
        dense0 = &layers[0];
        act_fn = &layers[1];
        dense1 = &layers[2];
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters) override
    {
        twopc_protocols::ShareTensor<T> hidden_states = (*this->dense0)(x);
        hidden_states = (*this->act_fn)(hidden_states, parameters);
        return (*this->dense1)(hidden_states);
    }
};

template <typename T>
class SecEncoderLayer : public Layers<T>
{
public:
    Layers<T> *attention, *LayerNorm1, *ffn, *LayerNorm2;

    SecEncoderLayer(TransformerConfig &config) : Layers<T>("EncoderLayer")
    {
        attention = new SecAttention<T>(config);
        LayerNorm1 = new SecLayerNorm<T>(config.hidden_size);
        ffn = new SecFFN<T>(config);
        LayerNorm2 = new SecLayerNorm<T>(config.hidden_size);
    }

    SecEncoderLayer(TransformerConfig &config,
                    std::vector<Layers<T>> layers) : Layers<T>("EncoderLayer")
    {
        attention = &layers[0];
        LayerNorm1 = &layers[1];
        ffn = &layers[2];
        LayerNorm2 = &layers[3];
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters) override
    {
        twopc_protocols::ShareTensor<T> self_attention_output = (*this->attention)(x, parameters);
        twopc_protocols::add(x, self_attention_output, self_attention_output);
        twopc_protocols::ShareTensor<T> attention_output = (*this->LayerNorm1)(self_attention_output, parameters);

        twopc_protocols::ShareTensor<T> ffn_output = (*this->ffn)(attention_output, parameters);
        twopc_protocols::add(attention_output, ffn_output, ffn_output);
        return (*this->LayerNorm2)(ffn_output, parameters);
    }
};

template <typename T>
class SecDecoderLayer : public Layers<T>
{
public:
    Layers<T> *masked_attention, *LayerNorm0, *attention, *LayerNorm1, *ffn, *LayerNorm2;

    SecDecoderLayer(TransformerConfig &config) : Layers<T>("DecoderLayer")
    {
        masked_attention = new SecAttention<T>(config, true);
        LayerNorm0 = new SecLayerNorm<T>(config.hidden_size);
        attention = new SecAttention<T>(config);
        LayerNorm1 = new SecLayerNorm<T>(config.hidden_size);
        ffn = new SecFFN<T>(config);
        LayerNorm2 = new SecLayerNorm<T>(config.hidden_size);
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters) override
    {
        // TODO: input from previous layers
        twopc_protocols::ShareTensor<T> masked_attention_output = (*this->masked_attention)(x, parameters);
        twopc_protocols::add(x, masked_attention_output, masked_attention_output);
        twopc_protocols::ShareTensor<T> masked_attention_output_norm = (*this->LayerNorm0)(masked_attention_output, parameters);

        twopc_protocols::ShareTensor<T> attention_output = (*this->attention)(masked_attention_output_norm, parameters);
        twopc_protocols::add(masked_attention_output_norm, attention_output, attention_output);
        twopc_protocols::ShareTensor<T> attention_output_norm = (*this->LayerNorm1)(attention_output, parameters);

        twopc_protocols::ShareTensor<T> ffn_output = (*this->ffn)(attention_output_norm, parameters);
        twopc_protocols::add(attention_output_norm, ffn_output, ffn_output);
        return (*this->LayerNorm2)(ffn_output, parameters);
    }
};

template <typename T>
class SecGPT2Layer : public Layers<T>
{
public:
    Layers<T> *LayerNorm0, *masked_attention, *LayerNorm1, *ffn;

    SecGPT2Layer(TransformerConfig &config) : Layers<T>("GPT2Layer")
    {
        LayerNorm0 = new SecLayerNorm<T>(config.hidden_size);
        masked_attention = new SecGPT2Attention<T>(config);
        LayerNorm1 = new SecLayerNorm<T>(config.hidden_size);
        ffn = new SecFFN<T>(config);
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters) override
    {
        twopc_protocols::ShareTensor<T> ln0_out = (*this->LayerNorm0)(x, parameters);
        twopc_protocols::ShareTensor<T> attn_out = (*this->masked_attention)(ln0_out, parameters);
        twopc_protocols::add(x, attn_out, attn_out);
        twopc_protocols::ShareTensor<T> ln1_out = (*this->LayerNorm1)(attn_out, parameters);
        twopc_protocols::ShareTensor<T> ffn_out = (*this->ffn)(ln1_out, parameters);
        twopc_protocols::add(attn_out, ffn_out, ffn_out);
        return ffn_out;
    }
};

template <typename T>
class SecEncoder : public Layers<T>
{
public:
    TransformerConfig config;
    std::vector<SecEncoderLayer<T>> layers;

    SecEncoder(TransformerConfig &config) : Layers<T>("Encoder"), config(config)
    {
        for (int i = 0; i < config.num_layers; i++)
        {
            layers.emplace_back(config);
        }
    }

    SecEncoder(TransformerConfig &config,
               std::vector<Layers<T>> all_layers) : Layers<T>("Encoder"),
                                                    config(config)
    {
        always_assert(config.num_layers == all_layers.size());
        for (int i = 0; i < config.num_layers; i++)
        {
            layers.emplace_back(config, all_layers[i]);
        }
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters) override
    {
        twopc_protocols::ShareTensor<T> hidden_states(x.shape);
        hidden_states = x;
        for (int i = 0; i < this->config.num_layers; i++)
        {
            hidden_states = this->layers[i](hidden_states, parameters);
        }
        return hidden_states;
    }
};

template <typename T>
class SecDecoder : public Layers<T>
{
public:
    TransformerConfig config;
    std::vector<SecDecoderLayer<T>> layers;

    SecDecoder(TransformerConfig &config) : Layers<T>("Decoder"), config(config)
    {
        for (int i = 0; i < config.num_layers; i++)
        {
            layers.emplace_back(config);
        }
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters) override
    {
        twopc_protocols::ShareTensor<T> hidden_states(x.shape);
        hidden_states = x;
        for (int i = 0; i < this->config.num_layers; i++)
        {
            hidden_states = this->layers[i](hidden_states, parameters);
        }
        return hidden_states;
    }
};

template <typename T>
class SecGPT2 : public Layers<T>
{
public:
    TransformerConfig config;
    std::vector<SecGPT2Layer<T>> layers;
    Layers<T> *ln_f;

    SecGPT2(TransformerConfig &config) : Layers<T>("GPT2"), config(config)
    {
        for (int i = 0; i < config.num_layers; i++)
        {
            layers.emplace_back(config);
        }
        ln_f = new SecLayerNorm<T>(config.hidden_size);
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters) override
    {
        twopc_protocols::ShareTensor<T> hidden_states(x.shape);
        hidden_states = x;
        for (int i = 0; i < this->config.num_layers; i++)
        {
            hidden_states = this->layers[i](hidden_states, parameters);
        }
        return (*this->ln_f)(hidden_states, parameters);
    }
};

template <typename T>
class SecAlexNet : public Layers<T>
{
    // AlexNet for CIFAR 10, the same arichtecture as Falcon
public:
    Layers<T> *conv1, *pool1, *relu1, *conv2, *pool2, *relu2, *conv3, *relu3, *conv4, *relu4, *conv5, *relu5, *fc1, *relu6, *fc2, *relu7, *fc3, *relu8;

    SecAlexNet() : Layers<T>("AlexNet")
    {
        conv1 = new SecConv<T>(3, 96, 11, 4, 9);
        pool1 = new SecMaxPool<T>(3, 2);
        relu1 = new SecReLU<T>();

        conv2 = new SecConv<T>(96, 256, 5, 1, 1);
        pool2 = new SecMaxPool<T>(3, 2);
        relu2 = new SecReLU<T>();

        conv3 = new SecConv<T>(256, 384, 3, 1, 1);
        relu3 = new SecReLU<T>();

        conv4 = new SecConv<T>(384, 384, 3, 1, 1);
        relu4 = new SecReLU<T>();

        conv5 = new SecConv<T>(384, 256, 3, 1, 1);
        relu5 = new SecReLU<T>();

        fc1 = new SecLinear<T>(256, 256);
        relu6 = new SecReLU<T>();
        
        fc2 = new SecLinear<T>(256, 256);
        relu7 = new SecReLU<T>();

        fc3 = new SecLinear<T>(256, 10);
        relu8 = new SecReLU<T>();
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters) override
    {
        twopc_protocols::ShareTensor<T> res(x.shape);
        res = x;
        res = (*this->conv1)(res, parameters);
        res = (*this->pool1)(res, parameters);
        res = (*this->relu1)(res, parameters);

        res = (*this->conv2)(res, parameters);
        res = (*this->pool2)(res, parameters);
        res = (*this->relu2)(res, parameters);

        res = (*this->conv3)(res, parameters);
        res = (*this->relu3)(res, parameters);

        res = (*this->conv4)(res, parameters);
        res = (*this->relu4)(res, parameters);

        res = (*this->conv5)(res, parameters);
        res = (*this->relu5)(res, parameters);

        res.reshape({res.size() / (256), 256});

        res = (*this->fc1)(res, parameters);
        res = (*this->relu6)(res, parameters);
        res = (*this->fc2)(res, parameters);
        res = (*this->relu7)(res, parameters);
        res = (*this->fc3)(res, parameters);
        res = (*this->relu8)(res, parameters);
        return res;
    }
};

template <typename T>
class ShortCut : public Layers<T>
{
public:
    Layers<T> *conv, *bn;

    ShortCut(uint32_t in_channels, uint32_t out_channels, uint32_t expansion, uint32_t stride) : Layers<T>("Shortcut")
    {
        conv = new SecConv<T>(in_channels, out_channels * expansion, 1, stride, 0);
        bn = new SecBatchNorm<T>(out_channels * expansion);
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x) override
    {
        twopc_protocols::ShareTensor<T> res = (*this->conv)(x);
        return (*this->bn)(res);
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters) override
    {
        return forward(x);
    }
};

template <typename T>
class BottleNeck : public Layers<T>
{
public:
    Layers<T> *conv1, *bn1, *relu1, *conv2, *bn2, *relu2, *conv3, *bn3, *shortcut, *relu;
    uint32_t expansion = 4;
    
    BottleNeck(uint32_t in_channels, uint32_t out_channels, uint32_t stride = 1) : Layers<T>("BottleNeck")
    {
        conv1 = new SecConv<T>(in_channels, out_channels, 1);
        bn1 = new SecBatchNorm<T>(out_channels);
        relu1 = new SecReLU<T>();
        conv2 = new SecConv<T>(out_channels, out_channels, 3, stride, 1);
        bn2 = new SecBatchNorm<T>(out_channels);
        relu2 = new SecReLU<T>();
        conv3 = new SecConv<T>(out_channels, out_channels * expansion, 1);
        bn3 = new SecBatchNorm<T>(out_channels * expansion);

        relu = new SecReLU<T>();

        if (stride != 1 || in_channels != out_channels * expansion)
        {
            shortcut = new ShortCut<T>(in_channels, out_channels, expansion, stride);
        }
        else
        {
            shortcut = new Layers<T>();
        }
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters) override
    {
        twopc_protocols::ShareTensor<T> x_copy(x.shape);
        x_copy = x;
        twopc_protocols::ShareTensor<T> residual_res = (*this->conv1)(x);
        residual_res = (*this->bn1)(residual_res);
        residual_res = (*this->relu1)(residual_res, parameters);

        residual_res = (*this->conv2)(residual_res);
        residual_res = (*this->bn2)(residual_res);
        residual_res = (*this->relu2)(residual_res, parameters);

        residual_res = (*this->conv3)(residual_res);
        residual_res = (*this->bn3)(residual_res);

        twopc_protocols::ShareTensor<T> shortcut_res = (*this->shortcut)(x_copy);
        twopc_protocols::add(residual_res, shortcut_res, residual_res);

        return (*this->relu1)(residual_res, parameters);
    }
};

template <typename T>
class SecResNet50 : public Layers<T>
{
public:
    Layers<T> *conv, *bn, *relu, *pool;
    std::vector<BottleNeck<T>> conv2_x, conv3_x, conv4_x, conv5_x;
    Layers<T> *avg_pool, *fc;
    uint32_t in_channels = 64;
    uint32_t expansion = 4;

    std::vector<BottleNeck<T>> make_layer(uint32_t out_channels, uint32_t num_blocks, uint32_t stride)
    {
        std::vector<BottleNeck<T>> layers;

        for(int i = 0; i < num_blocks; i++)
        {
            if (i != 0)
            {
                stride = 1;
            }
            layers.emplace_back(this->in_channels, out_channels, stride);
            this->in_channels = out_channels * expansion;
        }

        return layers;
    }

    SecResNet50(uint32_t num_classes = 100) : Layers<T>("resnet")
    {
        conv = new SecConv<T>(3, 64, 7, 2, 3);
        bn = new SecBatchNorm<T>(64);
        relu = new SecReLU<T>();
        pool = new SecMaxPool<T>(3, 2, 1);

        conv2_x = make_layer(64, 3, 1);
        conv3_x = make_layer(128, 4, 2);
        conv4_x = make_layer(256, 6, 2);
        conv5_x = make_layer(512, 3, 2);

        avg_pool = new SecAdaptiveAvgPool<T>(1);
        fc = new SecLinear<T>(512 * expansion, num_classes);
    }

    twopc_protocols::ShareTensor<T> forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters) override
    {
        twopc_protocols::ShareTensor<T> output = (*this->conv)(x);
        output = (*this->bn)(output);
        output = (*this->relu)(output, parameters);
        output = (*this->pool)(output, parameters);

        for(int i = 0; i < conv2_x.size(); i++)
        {
            output = this->conv2_x[i](output, parameters);
        }

        for(int i = 0; i < conv3_x.size(); i++)
        {
            output = this->conv3_x[i](output, parameters);
        }

        for(int i = 0; i < conv4_x.size(); i++)
        {
            output = this->conv4_x[i](output, parameters);
        }

        for(int i = 0; i < conv5_x.size(); i++)
        {
            output = this->conv5_x[i](output, parameters);
        }

        output = (*this->avg_pool)(output);
        output.reshape({output.shape[0], output.size() / output.shape[0]});
        return (*this->fc)(output);
    }

};

class SecMixedLayerNorm : public Layers<uint64_t>
{
public:
    uint32_t normalized_shape;
    twopc_protocols::ShareTensor<uint64_t> weight;
    twopc_protocols::ShareTensor<uint64_t> bias;

    SecMixedLayerNorm(uint32_t normalized_shape) : Layers<uint64_t>("LayerNorm"),
                                                   normalized_shape(normalized_shape)
    {
        weight.allocate({normalized_shape});
        bias.allocate({normalized_shape});
        weight.zeros();
        bias.zeros();
        this->parameters["weight"] = weight;
        this->parameters["bias"] = bias;

        this->need_parameters = true;
    }

    SecMixedLayerNorm(twopc_protocols::ShareTensor<uint64_t> &weight, twopc_protocols::ShareTensor<uint64_t> &bias) : Layers<uint64_t>("LayerNorm")
    {
        normalized_shape = weight.shape[0];
        weight.allocate({normalized_shape});
        bias.allocate({normalized_shape});
        this->weight = weight;
        this->bias = bias;
        this->parameters["weight"] = weight;
        this->parameters["bias"] = bias;
    }

    void set_weight(twopc_protocols::ShareTensor<uint64_t> &weight)
    {
        always_assert(this->normalized_shape == weight.shape[0]);
        this->weight = weight;
        this->parameters["weight"] = weight;
    }

    void set_bias(twopc_protocols::ShareTensor<uint64_t> &bias)
    {
        always_assert(this->normalized_shape == bias.shape[0]);
        this->bias = bias;
        this->parameters["bias"] = bias;
    }

    void init_param()
    {
        weight = this->parameters["weight"];
        bias = this->parameters["bias"];
    }

    twopc_protocols::ShareTensor<uint64_t> forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters) override
    {
        perf_stats::Scope perf_scope("SecMixedLayerNorm");
        twopc_protocols::ShareTensor<uint64_t> res(x.shape);

        std::vector<uint32_t> sum_shape = x.shape;
        uint32_t dim_size = sum_shape.back();
        always_assert(dim_size == this->normalized_shape); // can only normalize the last dimension
        sum_shape.pop_back();

        twopc_protocols::ShareTensor<uint64_t> x_sum(sum_shape);
        const uint32_t sum_size = x_sum.size();
#pragma omp parallel for if (utils::should_parallelize(static_cast<uint64_t>(sum_size) * dim_size))
        for (int i = 0; i < sum_size; i++)
        {
            x_sum.share.data[i] = 0;
            for (int j = 0; j < dim_size; j++)
            {
                x_sum.share.data[i] += x.share.data[i * dim_size + j];
            }
        }

        twopc_protocols::truncate(x_sum, normalized_shape, IS_MALICIOUS); // calculate mean

        twopc_protocols::ShareTensor<uint64_t> z(x.shape);
#pragma omp parallel for if (utils::should_parallelize(static_cast<uint64_t>(sum_size) * dim_size))
        for (int i = 0; i < sum_size; i++)
        {
            for (int j = 0; j < dim_size; j++)
            {
                z.share.data[i * dim_size + j] = x.share.data[i * dim_size + j] - x_sum.share.data[i];
            }
        }

        twopc_protocols::ShareTensor<uint64_t> z_square(x.shape), square_sum(sum_shape);
        twopc_protocols::square(z, z_square, true, IS_MALICIOUS); // calculate z^2

#pragma omp parallel for if (utils::should_parallelize(static_cast<uint64_t>(sum_size) * dim_size))
        for (int i = 0; i < sum_size; i++)
        {
            square_sum.share.data[i] = 0;
            for (int j = 0; j < dim_size; j++)
            {
                square_sum.share.data[i] += z_square.share.data[i * dim_size + j];
            }
        }

        twopc_protocols::truncate(square_sum, normalized_shape, IS_MALICIOUS); // calculate variance

        twopc_protocols::ShareTensor<uint64_t> rsqrt_var(sum_shape), broadcast_rsqrt_var(x.shape), gamma(x.shape);
        twopc_protocols::rsqrt(square_sum, rsqrt_var, parameters, IS_MALICIOUS); // calculate 1 / sqrt(var)

    #pragma omp parallel for if (utils::should_parallelize(static_cast<uint64_t>(sum_size) * dim_size))
        for (int i = 0; i < sum_size; i++)
        {
            for (int j = 0; j < dim_size; j++)
            {
                broadcast_rsqrt_var.share.data[i * dim_size + j] = rsqrt_var.share.data[i];

                gamma.share.data[i * dim_size + j] = this->weight.share.data[j];
            }
        }

        twopc_protocols::ShareTensor<uint64_t> z_norm(x.shape), z_norm_weighted(x.shape);
        twopc_protocols::mul(z, broadcast_rsqrt_var, z_norm, true, IS_MALICIOUS); // calculate z_norm
        twopc_protocols::mul(z_norm, gamma, z_norm_weighted, true, IS_MALICIOUS); // calculate gamma * z_norm

        // calculate gamma * z_norm + beta
    #pragma omp parallel for if (utils::should_parallelize(static_cast<uint64_t>(sum_size) * dim_size))
        for (int i = 0; i < sum_size; i++)
        {
            for (int j = 0; j < dim_size; j++)
            {
                res.share.data[i * dim_size + j] = z_norm_weighted.share.data[i * dim_size + j] + this->bias.share.data[j];
            }
        }

        return res;
    }
};

class SecMixedRMSNorm : public Layers<uint64_t>
{
public:
    uint32_t normalized_shape;
    twopc_protocols::ShareTensor<uint64_t> weight;

    SecMixedRMSNorm(uint32_t normalized_shape) : Layers<uint64_t>("RMSNorm"), normalized_shape(normalized_shape)
    {
        weight.allocate({normalized_shape});
        weight.zeros();
        this->parameters["weight"] = weight;
        this->need_parameters = true;
    }

    SecMixedRMSNorm(twopc_protocols::ShareTensor<uint64_t> &weight) : Layers<uint64_t>("RMSNorm")
    {
        normalized_shape = weight.shape[0];
        this->weight = weight;
        this->parameters["weight"] = weight;
    }

    void set_weight(twopc_protocols::ShareTensor<uint64_t> &weight)
    {
        always_assert(this->normalized_shape == weight.shape[0]);
        this->weight = weight;
        this->parameters["weight"] = weight;
    }

    void init_param()
    {
        weight = this->parameters["weight"];
    }

    twopc_protocols::ShareTensor<uint64_t> forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters) override
    {
        perf_stats::Scope perf_scope("SecMixedRMSNorm");
        const auto profile_begin = std::chrono::steady_clock::now();
        twopc_protocols::ShareTensor<uint64_t> res(x.shape);
        twopc_protocols::rms_norm(x, weight, res, parameters, IS_MALICIOUS);
        const uint64_t elapsed_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - profile_begin)
                .count());
        llama_profile::add_rmsnorm_ns(elapsed_ns);
        return res;
    }
};

class SecMixedSelfAttention : public Layers<uint64_t>
{
public:
    uint32_t num_heads;
    uint32_t head_size;
    uint32_t all_head_size;
    Layers<uint64_t> *query, *key, *value;
    Layers<uint64_t> *softmax;

    SecMixedSelfAttention(TransformerConfig &config, const bool masked = false) : Layers<uint64_t>("SelfAttention"),
                                                                                  num_heads(config.num_attention_heads),
                                                                                  head_size(
                                                                                      config.hidden_size / config.num_attention_heads),
                                                                                  all_head_size(num_heads * head_size)
    {
        query = new SecLinear<uint64_t>(config.hidden_size, all_head_size);
        key = new SecLinear<uint64_t>(config.hidden_size, all_head_size);
        value = new SecLinear<uint64_t>(config.hidden_size, all_head_size);
        softmax = new SecSoftmax<uint64_t>(masked);
    }

    twopc_protocols::ShareTensor<uint64_t> forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters) override
    {
        twopc_protocols::ShareTensor<uint64_t> mixed_query, mixed_key, mixed_value;
        {
            perf_stats::Scope perf_scope("SecMixedSelfAttention::qkv_proj");
            mixed_query = (*this->query)(x);
            mixed_key = (*this->key)(x);
            mixed_value = (*this->value)(x);
        }

        std::vector<uint32_t> new_shape = x.shape;
        new_shape.pop_back();
        new_shape.push_back(this->num_heads);
        new_shape.push_back(this->head_size);

        mixed_query.reshape(new_shape);
        mixed_key.reshape(new_shape);
        mixed_value.reshape(new_shape);

        twopc_protocols::ShareTensor<uint64_t> query_layer = mixed_query.permute({0, 2, 1, 3});
        twopc_protocols::ShareTensor<uint64_t> key_layer = mixed_key.permute({0, 2, 3, 1});
        twopc_protocols::ShareTensor<uint64_t> value_layer = mixed_value.permute({0, 2, 1, 3});

        twopc_protocols::ShareTensor<uint64_t> attention_scores(
            {query_layer.shape[0], query_layer.shape[1], query_layer.shape[2], key_layer.shape[3]});
        {
            perf_stats::Scope perf_scope("SecMixedSelfAttention::qk_matmul");
            twopc_protocols::matMul(query_layer, key_layer, attention_scores, true, IS_MALICIOUS);
        }
        twopc_protocols::truncate(attention_scores, (uint32_t)sqrt((double)this->head_size), IS_MALICIOUS);

        twopc_protocols::ShareTensor<uint64_t> attention_probs = (*this->softmax)(attention_scores, parameters);

        twopc_protocols::ShareTensor<uint64_t> context_layer({attention_probs.shape[0], attention_probs.shape[1], attention_probs.shape[2],
                                           value_layer.shape[3]});
        {
            perf_stats::Scope perf_scope("SecMixedSelfAttention::av_matmul");
            twopc_protocols::matMul(attention_probs, value_layer, context_layer, true, IS_MALICIOUS);
        }

        twopc_protocols::ShareTensor<uint64_t> res = context_layer.permute({0, 2, 1, 3});
        std::vector<uint32_t> res_shape = res.shape;
        res_shape[res_shape.size() - 2] = this->all_head_size;
        res_shape.pop_back();
        res.reshape(res_shape);

        return res;
    }
};

class SecMixedLlama2SelfAttention : public Layers<uint64_t>
{
public:
    uint32_t num_heads;
    uint32_t num_key_value_heads;
    uint32_t head_size;
    uint32_t all_head_size;
    uint32_t kv_all_head_size;
    Layers<uint64_t> *qkv_proj;
    Layers<uint64_t> *softmax;

    SecMixedLlama2SelfAttention(TransformerConfig &config, const bool masked = false) : Layers<uint64_t>("Llama2SelfAttention"),
                                                                                        num_heads(config.num_attention_heads),
                                                                                        num_key_value_heads(config.num_key_value_heads),
                                                                                        head_size(config.hidden_size / config.num_attention_heads),
                                                                                        all_head_size(num_heads * head_size),
                                                                                        kv_all_head_size(num_key_value_heads * head_size)
    {
        always_assert(num_heads % num_key_value_heads == 0);
        qkv_proj = new SecLinear<uint64_t>(config.hidden_size, all_head_size + 2 * kv_all_head_size);
        softmax = new SecSoftmax<uint64_t>(masked);
    }

    twopc_protocols::ShareTensor<uint64_t> forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters) override
    {
        twopc_protocols::ShareTensor<uint64_t> qkv = (*this->qkv_proj)(x);
        twopc_protocols::ShareTensor<uint64_t> mixed_query(x.shape);
        twopc_protocols::ShareTensor<uint64_t> mixed_key({x.shape[0], x.shape[1], this->kv_all_head_size});
        twopc_protocols::ShareTensor<uint64_t> mixed_value({x.shape[0], x.shape[1], this->kv_all_head_size});

        const uint32_t seq_batch = x.shape[0] * x.shape[1];
        const uint32_t q_dim = this->all_head_size;
        const uint32_t k_dim = this->kv_all_head_size;
        const uint32_t qkv_dim = qkv.shape.back();

#pragma omp parallel for
        for (uint32_t i = 0; i < seq_batch; i++)
        {
            const uint32_t qkv_base = i * qkv_dim;
            const uint32_t q_base = i * q_dim;
            const uint32_t k_base = i * k_dim;

            for (uint32_t j = 0; j < q_dim; j++)
            {
                mixed_query.share.data[q_base + j] = qkv.share.data[qkv_base + j];
            }
            for (uint32_t j = 0; j < k_dim; j++)
            {
                mixed_key.share.data[k_base + j] = qkv.share.data[qkv_base + q_dim + j];
                mixed_value.share.data[k_base + j] = qkv.share.data[qkv_base + q_dim + k_dim + j];
            }
        }

        std::vector<uint32_t> q_shape = x.shape;
        q_shape.pop_back();
        q_shape.push_back(this->num_heads);
        q_shape.push_back(this->head_size);
        mixed_query.reshape(q_shape);

        std::vector<uint32_t> kv_shape = x.shape;
        kv_shape.pop_back();
        kv_shape.push_back(this->num_key_value_heads);
        kv_shape.push_back(this->head_size);
        mixed_key.reshape(kv_shape);
        mixed_value.reshape(kv_shape);

        twopc_protocols::ShareTensor<uint64_t> query_layer = mixed_query.permute({0, 2, 1, 3});
        twopc_protocols::ShareTensor<uint64_t> key_heads = mixed_key.permute({0, 2, 1, 3});
        twopc_protocols::ShareTensor<uint64_t> value_heads = mixed_value.permute({0, 2, 1, 3});

        llama_utils::apply_rope_inplace(query_layer);
        llama_utils::apply_rope_inplace(key_heads);

        twopc_protocols::ShareTensor<uint64_t> key_layer = key_heads.permute({0, 1, 3, 2});
        twopc_protocols::ShareTensor<uint64_t> key_for_attn = key_layer;
        twopc_protocols::ShareTensor<uint64_t> value_for_attn = value_heads;

        if (this->num_key_value_heads != this->num_heads)
        {
            twopc_protocols::ShareTensor<uint64_t> expanded_key({key_layer.shape[0], this->num_heads, key_layer.shape[2], key_layer.shape[3]});
            twopc_protocols::ShareTensor<uint64_t> expanded_value({value_heads.shape[0], this->num_heads, value_heads.shape[2], value_heads.shape[3]});
            llama_utils::expand_kv_heads(key_layer, expanded_key);
            llama_utils::expand_kv_heads(value_heads, expanded_value);
            key_for_attn = expanded_key;
            value_for_attn = expanded_value;
        }

        twopc_protocols::ShareTensor<uint64_t> attention_scores(
            {query_layer.shape[0], query_layer.shape[1], query_layer.shape[2], key_for_attn.shape[3]});
        twopc_protocols::matMul(query_layer, key_for_attn, attention_scores, true, IS_MALICIOUS);
        twopc_protocols::truncate(attention_scores, (uint32_t)sqrt((double)this->head_size), IS_MALICIOUS);

        twopc_protocols::ShareTensor<uint64_t> attention_probs = (*this->softmax)(attention_scores, parameters);

        twopc_protocols::ShareTensor<uint64_t> context_layer({attention_probs.shape[0], attention_probs.shape[1], attention_probs.shape[2],
                                                               value_for_attn.shape[3]});
        twopc_protocols::matMul(attention_probs, value_for_attn, context_layer, true, IS_MALICIOUS);

        twopc_protocols::ShareTensor<uint64_t> res = context_layer.permute({0, 2, 1, 3});
        std::vector<uint32_t> res_shape = res.shape;
        res_shape[res_shape.size() - 2] = this->all_head_size;
        res_shape.pop_back();
        res.reshape(res_shape);

        return res;
    }
};

class SecMixedLlama2Attention : public Layers<uint64_t>
{
public:
    Layers<uint64_t> *self;
    Layers<uint64_t> *dense;

    SecMixedLlama2Attention(TransformerConfig &config, const bool masked = false) : Layers<uint64_t>("Llama2Attention")
    {
        self = new SecMixedLlama2SelfAttention(config, masked);
        dense = new SecLinear<uint64_t>(config.hidden_size, config.hidden_size);
    }

    twopc_protocols::ShareTensor<uint64_t> forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters) override
    {
        twopc_protocols::ShareTensor<uint64_t> self_outputs = (*this->self)(x, parameters);
        perf_stats::Scope perf_scope("SecMixedAttention::out_proj");
        return (*this->dense)(self_outputs);
    }
};

class SecMixedAttention : public Layers<uint64_t>
{
public:
    Layers<uint64_t> *self;
    Layers<uint64_t> *dense;

    SecMixedAttention(TransformerConfig &config, const bool masked = false) : Layers<uint64_t>(
                                                                                  "Attention")
    {
        self = new SecMixedSelfAttention(config, masked);
        dense = new SecLinear<uint64_t>(config.hidden_size, config.hidden_size);
    }

    twopc_protocols::ShareTensor<uint64_t> forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters) override
    {
        twopc_protocols::ShareTensor<uint64_t> self_outputs = (*this->self)(x, parameters);
        return (*this->dense)(self_outputs);
    }
};

class SecMixedGPT2Attention : public Layers<uint64_t>
{
public:
    uint32_t num_heads;
    uint32_t head_size;
    uint32_t all_head_size;
    Layers<uint64_t> *c_attn, *c_proj;
    Layers<uint64_t> *softmax;

    SecMixedGPT2Attention(TransformerConfig &config) : Layers<uint64_t>("GPT2Attention"),
                                                       num_heads(config.num_attention_heads),
                                                       head_size(config.hidden_size / config.num_attention_heads),
                                                       all_head_size(num_heads * head_size)
    {
        c_attn = new SecLinear<uint64_t>(config.hidden_size, 3 * all_head_size);
        softmax = new SecSoftmax<uint64_t>(true);
        c_proj = new SecLinear<uint64_t>(config.hidden_size, all_head_size);
    }

    twopc_protocols::ShareTensor<uint64_t> forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters) override
    {
        twopc_protocols::ShareTensor<uint64_t> qkv;
        {
            perf_stats::Scope perf_scope("SecMixedGPT2Attention::qkv_proj");
            qkv = (*this->c_attn)(x);
        }
        twopc_protocols::ShareTensor<uint64_t> mixed_query(x.shape), mixed_key(x.shape), mixed_value(x.shape);

        uint32_t size = x.size(), last_dim = x.shape[x.shape.size() - 1];
        uint32_t other_dim = size / last_dim;
        // split, dim = -1
        uint32_t qkv_last_dim = qkv.shape[qkv.shape.size() - 1];
        uint32_t index, qkv_index;
        for (int i = 0; i < other_dim; i++)
        {
            for (int j = 0; j < last_dim; j++)
            {
                index = i * last_dim + j;
                qkv_index = i * qkv_last_dim + j;

                mixed_query.share.data[index] = qkv.share.data[qkv_index];

                mixed_key.share.data[index] = qkv.share.data[qkv_index + last_dim];

                mixed_value.share.data[index] = qkv.share.data[qkv_index + last_dim * 2];
            }
        }

        std::vector<uint32_t> new_shape = x.shape;
        new_shape.pop_back();
        new_shape.push_back(this->num_heads);
        new_shape.push_back(this->head_size);

        mixed_query.reshape(new_shape);
        mixed_key.reshape(new_shape);
        mixed_value.reshape(new_shape);

        twopc_protocols::ShareTensor<uint64_t> query_layer = mixed_query.permute({0, 2, 1, 3});
        twopc_protocols::ShareTensor<uint64_t> key_layer = mixed_key.permute({0, 2, 3, 1});
        twopc_protocols::ShareTensor<uint64_t> value_layer = mixed_value.permute({0, 2, 1, 3});

        twopc_protocols::ShareTensor<uint64_t> attention_scores(
            {query_layer.shape[0], query_layer.shape[1], query_layer.shape[2], key_layer.shape[3]});
        {
            perf_stats::Scope perf_scope("SecMixedGPT2Attention::qk_matmul");
            twopc_protocols::matMul(query_layer, key_layer, attention_scores, true, IS_MALICIOUS);
        }
        twopc_protocols::truncate(attention_scores, (uint32_t)sqrt((double)this->head_size), IS_MALICIOUS);

        twopc_protocols::ShareTensor<uint64_t> attention_probs = (*this->softmax)(attention_scores, parameters);

        twopc_protocols::ShareTensor<uint64_t> context_layer({attention_probs.shape[0], attention_probs.shape[1], attention_probs.shape[2],
                                           value_layer.shape[3]});
        {
            perf_stats::Scope perf_scope("SecMixedGPT2Attention::av_matmul");
            twopc_protocols::matMul(attention_probs, value_layer, context_layer, true, IS_MALICIOUS);
        }

        twopc_protocols::ShareTensor<uint64_t> attn_output = context_layer.permute({0, 2, 1, 3});
        std::vector<uint32_t> output_shape = attn_output.shape;
        output_shape[output_shape.size() - 2] = this->all_head_size;
        output_shape.pop_back();
        attn_output.reshape(output_shape);

        perf_stats::Scope perf_scope("SecMixedGPT2Attention::out_proj");
        return (*this->c_proj)(attn_output);
    }
};

class SecMixedGroupedQueryAttention : public Layers<uint64_t>
{
public:
    uint32_t num_heads;
    uint32_t num_key_value_heads;
    uint32_t head_size;
    uint32_t all_head_size;
    uint32_t key_value_size;
    Layers<uint64_t> *qkv_proj, *o_proj;

    SecMixedGroupedQueryAttention(TransformerConfig &config) : Layers<uint64_t>("GroupedMultiQueryAttention"),
                                                               num_heads(config.num_attention_heads),
                                                               num_key_value_heads(config.num_key_value_heads),
                                                               head_size(config.hidden_size / config.num_attention_heads),
                                                               all_head_size(num_heads * head_size),
                                                               key_value_size(num_key_value_heads * head_size)
    {
        qkv_proj = new SecLinear<uint64_t>(config.hidden_size, all_head_size + 2 * key_value_size);
        o_proj = new SecLinear<uint64_t>(config.hidden_size, config.hidden_size);
    }

    twopc_protocols::ShareTensor<uint64_t> forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters) override
    {
        perf_stats::Scope perf_scope("SecMixedGroupedQueryAttention");
        const auto profile_begin = std::chrono::steady_clock::now();
        twopc_protocols::ShareTensor<uint64_t> query(x.shape), key({x.shape[0], x.shape[1], this->key_value_size}), value({x.shape[0], x.shape[1], this->key_value_size});
        twopc_protocols::ShareTensor<uint64_t> qkv;
        {
            perf_stats::Scope projection_scope("SecMixedGroupedQueryAttention::qkv_proj");
            qkv = (*this->qkv_proj)(x);
        }
        const uint32_t seq_batch = x.shape[0] * x.shape[1];
        const uint32_t q_dim = this->all_head_size;
        const uint32_t k_dim = this->key_value_size;
        const uint32_t qkv_dim = qkv.shape.back();

#pragma omp parallel for if (layer_parallel::should_parallelize(static_cast<uint64_t>(seq_batch) * q_dim))
        for (uint32_t i = 0; i < seq_batch; i++)
        {
            const uint32_t qkv_base = i * qkv_dim;
            const uint32_t q_base = i * q_dim;
            const uint32_t k_base = i * k_dim;

            for (uint32_t j = 0; j < q_dim; j++)
            {
                query.share.data[q_base + j] = qkv.share.data[qkv_base + j];
            }
            for (uint32_t j = 0; j < k_dim; j++)
            {
                key.share.data[k_base + j] = qkv.share.data[qkv_base + q_dim + j];
                value.share.data[k_base + j] = qkv.share.data[qkv_base + q_dim + k_dim + j];
            }
        }

        twopc_protocols::ShareTensor<uint64_t> attn_output(x.shape);
        twopc_protocols::grouped_multi_query_attention(query, key, value, attn_output, parameters,
                                                       num_heads, num_key_value_heads, true, IS_MALICIOUS);
        twopc_protocols::ShareTensor<uint64_t> output = attn_output;
        {
            perf_stats::Scope projection_scope("SecMixedGroupedQueryAttention::out_proj");
            output = (*this->o_proj)(attn_output);
        }
        const uint64_t elapsed_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - profile_begin)
                .count());
        llama_profile::add_attention_ns(elapsed_ns);
        return output;
    }
};

class SecMixedFFN : public Layers<uint64_t>
{
public:
    Layers<uint64_t> *dense0, *dense1;
    Layers<uint64_t> *act_fn;

    SecMixedFFN(TransformerConfig &config) : Layers<uint64_t>("FFN")
    {
        dense0 = new SecLinear<uint64_t>(config.hidden_size, config.intermediate_size);
        if (config.isReLUAct)
        {
            act_fn = new SecReLU<uint64_t>();
        }
        else
        {
            act_fn = new SecGELU<uint64_t>();
        }
        dense1 = new SecLinear<uint64_t>(config.intermediate_size, config.hidden_size);
    }

    twopc_protocols::ShareTensor<uint64_t> forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters) override
    {
        twopc_protocols::ShareTensor<uint64_t> hidden_states;
        {
            perf_stats::Scope perf_scope("SecMixedFFN::dense0");
            hidden_states = (*this->dense0)(x);
        }
        {
            perf_stats::Scope perf_scope("SecMixedFFN::activation");
            hidden_states = (*this->act_fn)(hidden_states, parameters);
        }
        perf_stats::Scope perf_scope("SecMixedFFN::dense1");
        return (*this->dense1)(hidden_states);
    }
};

class SecMixedSwiGLU : public Layers<uint64_t>
{
public:
    SecLinear<uint64_t> *gate_proj, *up_proj, *down_proj;

    SecMixedSwiGLU(TransformerConfig &config) : Layers<uint64_t>("SwiGLU")
    {
        gate_proj = new SecLinear<uint64_t>(config.hidden_size, config.intermediate_size);
        up_proj = new SecLinear<uint64_t>(config.hidden_size, config.intermediate_size);
        down_proj = new SecLinear<uint64_t>(config.intermediate_size, config.hidden_size);
    }

    twopc_protocols::ShareTensor<uint64_t> forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters) override
    {
        perf_stats::Scope perf_scope("SecMixedSwiGLU");
        const auto profile_begin = std::chrono::steady_clock::now();
        twopc_protocols::ShareTensor<uint64_t> gate, up;
        {
            perf_stats::Scope stage_scope("SecMixedSwiGLU::gate_up_proj");
            if (!IS_MALICIOUS)
            {
                this->gate_proj->load_param();
                this->up_proj->load_param();
                fused_linear_pair_forward(x, *this->gate_proj, *this->up_proj, gate, up);
            }
            else
            {
                const bool parallel_projections = layer_parallel::should_parallelize_tasks(x.size(), 1, 2);
                if (parallel_projections)
                {
                    const int thread_limit = layer_parallel::concurrent_task_thread_limit(2);
                    std::thread gate_thread([&]() {
                        twopc_protocols::parallel::ScopedOmpThreadLimit scoped_limit(thread_limit);
                        gate = (*this->gate_proj)(x);
                    });

                    {
                        twopc_protocols::parallel::ScopedOmpThreadLimit scoped_limit(thread_limit);
                        up = (*this->up_proj)(x);
                    }

                    gate_thread.join();
                }
                else
                {
                    gate = (*this->gate_proj)(x);
                    up = (*this->up_proj)(x);
                }
            }
        }

        twopc_protocols::ShareTensor<uint64_t> fused(gate.shape);
        {
            perf_stats::Scope stage_scope("SecMixedSwiGLU::activation");
            twopc_protocols::swiglu(gate, up, fused, parameters, IS_MALICIOUS);
        }
        twopc_protocols::ShareTensor<uint64_t> output;
        {
            perf_stats::Scope stage_scope("SecMixedSwiGLU::down_proj");
            output = (*this->down_proj)(fused);
        }
        const uint64_t elapsed_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - profile_begin)
                .count());
        llama_profile::add_swiglu_ns(elapsed_ns);
        return output;
    }
};

class SecMixedEncoderLayer : public Layers<uint64_t>
{
public:
    Layers<uint64_t> *attention, *LayerNorm1, *LayerNorm2;
    Layers<uint64_t> *ffn;

    SecMixedEncoderLayer(TransformerConfig &config) : Layers<uint64_t>("EncoderLayer")
    {
        attention = new SecMixedAttention(config);
        LayerNorm1 = new SecMixedLayerNorm(config.hidden_size);
        ffn = new SecMixedFFN(config);
        LayerNorm2 = new SecMixedLayerNorm(config.hidden_size);
    }

    twopc_protocols::ShareTensor<uint64_t> forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters) override
    {
        twopc_protocols::ShareTensor<uint64_t> self_attention_output = (*this->attention)(x, parameters);
        twopc_protocols::add(x, self_attention_output, self_attention_output);
        twopc_protocols::ShareTensor<uint64_t> attention_output = (*this->LayerNorm1)(self_attention_output, parameters);

        twopc_protocols::ShareTensor<uint64_t> ffn_output = (*this->ffn)(attention_output, parameters);

        twopc_protocols::add(attention_output, ffn_output, ffn_output);
        return (*this->LayerNorm2)(ffn_output, parameters);
    }
};

class SecMixedDecoderLayer : public Layers<uint64_t>
{
public:
    Layers<uint64_t> *masked_attention, *LayerNorm0, *attention, *LayerNorm1, *LayerNorm2;
    Layers<uint64_t> *ffn;

    SecMixedDecoderLayer(TransformerConfig &config) : Layers<uint64_t>("DecoderLayer")
    {
        masked_attention = new SecMixedAttention(config, true);
        LayerNorm0 = new SecMixedLayerNorm(config.hidden_size);
        attention = new SecMixedAttention(config);
        LayerNorm1 = new SecMixedLayerNorm(config.hidden_size);
        ffn = new SecMixedFFN(config);
        LayerNorm2 = new SecMixedLayerNorm(config.hidden_size);
    }

    twopc_protocols::ShareTensor<uint64_t> forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters) override
    {
        // TODO: input from previous layers
        twopc_protocols::ShareTensor<uint64_t> masked_attention_output = (*this->masked_attention)(x, parameters);
        twopc_protocols::add(x, masked_attention_output, masked_attention_output);
        twopc_protocols::ShareTensor<uint64_t> masked_attention_output_norm = (*this->LayerNorm0)(masked_attention_output, parameters);

        twopc_protocols::ShareTensor<uint64_t> attention_output = (*this->attention)(masked_attention_output_norm, parameters);
        twopc_protocols::add(masked_attention_output_norm, attention_output, attention_output);
        twopc_protocols::ShareTensor<uint64_t> attention_output_norm = (*this->LayerNorm1)(attention_output, parameters);

        twopc_protocols::ShareTensor<uint64_t> ffn_output = (*this->ffn)(attention_output_norm, parameters);
        twopc_protocols::add(attention_output_norm, ffn_output, ffn_output);
        return (*this->LayerNorm2)(ffn_output, parameters);
    }
};

class SecMixedGPT2Layer : public Layers<uint64_t>
{
public:
    Layers<uint64_t> *LayerNorm0, *LayerNorm1;
    Layers<uint64_t> *masked_attention;
    Layers<uint64_t> *ffn;

    SecMixedGPT2Layer(TransformerConfig &config) : Layers<uint64_t>("GPT2Layer")
    {
        LayerNorm0 = new SecLayerNorm<uint64_t>(config.hidden_size);
        masked_attention = new SecMixedGPT2Attention(config);
        LayerNorm1 = new SecLayerNorm<uint64_t>(config.hidden_size);
        ffn = new SecMixedFFN(config);
    }

    twopc_protocols::ShareTensor<uint64_t> forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters) override
    {
        twopc_protocols::ShareTensor<uint64_t> ln0_out = (*this->LayerNorm0)(x, parameters);
        twopc_protocols::ShareTensor<uint64_t> attn_out = (*this->masked_attention)(ln0_out, parameters);
        twopc_protocols::add(x, attn_out, attn_out);
        twopc_protocols::ShareTensor<uint64_t> ln1_out = (*this->LayerNorm1)(attn_out, parameters);
        twopc_protocols::ShareTensor<uint64_t> ffn_out = (*this->ffn)(ln1_out, parameters);
        twopc_protocols::add(attn_out, ffn_out, ffn_out);
        return ffn_out;
    }
};

class SecMixedEncoder : public Layers<uint64_t>
{
public:
    TransformerConfig config;
    std::vector<SecMixedEncoderLayer> layers;

    SecMixedEncoder(TransformerConfig &config) : Layers<uint64_t>("Encoder"), config(config)
    {
        for (int i = 0; i < config.num_layers; i++)
        {
            layers.emplace_back(config);
        }
    }

    twopc_protocols::ShareTensor<uint64_t> forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters) override
    {
        twopc_protocols::ShareTensor<uint64_t> hidden_states(x.shape);
        hidden_states = x;
        for (int i = 0; i < this->config.num_layers; i++)
        {
            hidden_states = this->layers[i](hidden_states, parameters);
        }
        return hidden_states;
    }
};

class SecMixedDecoder : public Layers<uint64_t>
{
public:
    TransformerConfig config;
    std::vector<SecMixedDecoderLayer> layers;

    SecMixedDecoder(TransformerConfig &config) : Layers<uint64_t>("Decoder"), config(config)
    {
        for (int i = 0; i < config.num_layers; i++)
        {
            layers.emplace_back(config);
        }
    }

    twopc_protocols::ShareTensor<uint64_t> forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters) override
    {
        twopc_protocols::ShareTensor<uint64_t> hidden_states(x.shape);
        hidden_states = x;
        for (int i = 0; i < this->config.num_layers; i++)
        {
            hidden_states = this->layers[i](hidden_states, parameters);
        }
        return hidden_states;
    }
};

class SecMixedGPT2 : public Layers<uint64_t>
{
public:
    TransformerConfig config;
    std::vector<SecMixedGPT2Layer> layers;
    Layers<uint64_t> *ln_f;

    SecMixedGPT2(TransformerConfig &config) : Layers<uint64_t>("GPT2"), config(config)
    {
        for (int i = 0; i < config.num_layers; i++)
        {
            layers.emplace_back(config);
        }
        ln_f = new SecMixedLayerNorm(config.hidden_size);
    }

    twopc_protocols::ShareTensor<uint64_t> forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters) override
    {
        twopc_protocols::ShareTensor<uint64_t> hidden_states(x.shape);
        hidden_states = x;
        for (int i = 0; i < this->config.num_layers; i++)
        {
            hidden_states = this->layers[i](hidden_states, parameters);
        }
        return (*this->ln_f)(hidden_states, parameters);
    }
};

class SecMixedLlama2Layer : public Layers<uint64_t>
{
public:
    Layers<uint64_t> *input_norm, *attention, *post_attention_norm;
    Layers<uint64_t> *ffn;

    SecMixedLlama2Layer(TransformerConfig &config) : Layers<uint64_t>("Llama2Layer")
    {
        input_norm = new SecMixedRMSNorm(config.hidden_size);
        attention = new SecMixedGroupedQueryAttention(config);
        post_attention_norm = new SecMixedRMSNorm(config.hidden_size);
        ffn = new SecMixedSwiGLU(config);
    }

    twopc_protocols::ShareTensor<uint64_t> forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters) override
    {
        twopc_protocols::ShareTensor<uint64_t> norm_x = (*this->input_norm)(x, parameters);
        twopc_protocols::ShareTensor<uint64_t> attn_out = (*this->attention)(norm_x, parameters);
        twopc_protocols::add(x, attn_out, attn_out);

        twopc_protocols::ShareTensor<uint64_t> norm_attn = (*this->post_attention_norm)(attn_out, parameters);
        twopc_protocols::ShareTensor<uint64_t> ffn_out = (*this->ffn)(norm_attn, parameters);
        twopc_protocols::add(attn_out, ffn_out, ffn_out);
        return ffn_out;
    }
};

class SecMixedLlama2 : public Layers<uint64_t>
{
public:
    TransformerConfig config;
    std::vector<SecMixedLlama2Layer> layers;
    Layers<uint64_t> *norm;

    SecMixedLlama2(TransformerConfig &config) : Layers<uint64_t>("Llama2"), config(config)
    {
        for (int i = 0; i < config.num_layers; i++)
        {
            layers.emplace_back(config);
        }
        norm = new SecMixedRMSNorm(config.hidden_size);
    }

    twopc_protocols::ShareTensor<uint64_t> forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters) override
    {
        twopc_protocols::ShareTensor<uint64_t> hidden_states(x.shape);
        hidden_states = x;
        for (int i = 0; i < this->config.num_layers; i++)
        {
            hidden_states = this->layers[i](hidden_states, parameters);
        }
        return (*this->norm)(hidden_states, parameters);
    }
};

class SecBertClassifier : public Layers<uint64_t>
{
public:
    TransformerConfig config;
    Layers<uint64_t> *bert;
    Layers<uint64_t> *classifier;

    SecBertClassifier(TransformerConfig &config, int num_labels = 2) : Layers<uint64_t>("BertClassifier"), config(config)
    {
        bert = new SecMixedEncoder(config);
        classifier = new SecLinear<uint64_t>(config.hidden_size, num_labels);
    }

    twopc_protocols::ShareTensor<uint64_t> forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters) override
    {
        twopc_protocols::ShareTensor<uint64_t> bert_outputs(x.shape);
        bert_outputs = (*this->bert)(x, parameters);

        uint32_t batch_size = bert_outputs.shape[0];
        uint32_t seq_len = bert_outputs.shape[1];
        uint32_t hidden_size = config.hidden_size;
        uint32_t one_value_size = seq_len * hidden_size;
        twopc_protocols::ShareTensor<uint64_t> outputs({batch_size, hidden_size});
        for (int i = 0; i < batch_size; i++)
        {
            for (int j = 0; j < hidden_size; j++)
            {
                outputs.share.data[i * hidden_size + j] = bert_outputs.share.data[i * one_value_size + j];
            }
        }
        return (*this->classifier)(outputs);
    }

};

class SecBertClassifier64 : public Layers<uint64_t>
{
public:
    TransformerConfig config;
    Layers<uint64_t> *bert;
    Layers<uint64_t> *classifier;

    SecBertClassifier64(TransformerConfig &config, int num_labels = 2) : Layers<uint64_t>("BertClassifier"), config(config)
    {
        bert = new SecEncoder<uint64_t>(config);
        classifier = new SecLinear<uint64_t>(config.hidden_size, num_labels);
    }

    twopc_protocols::ShareTensor<uint64_t> forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters) override
    {
        twopc_protocols::ShareTensor<uint64_t> bert_outputs(x.shape);
        bert_outputs = (*this->bert)(x, parameters);

        uint32_t batch_size = bert_outputs.shape[0];
        uint32_t seq_len = bert_outputs.shape[1];
        uint32_t hidden_size = config.hidden_size;
        uint32_t one_value_size = seq_len * hidden_size;
        twopc_protocols::ShareTensor<uint64_t> outputs({batch_size, hidden_size});
        for (int i = 0; i < batch_size; i++)
        {
            for (int j = 0; j < hidden_size; j++)
            {
                outputs.share.data[i * hidden_size + j] = bert_outputs.share.data[i * one_value_size + j];
            }
        }
        return (*this->classifier)(outputs);
    }
};

class SecGPT2LMHead : public Layers<uint64_t>
{
public:
    TransformerConfig config;
    Layers<uint64_t> *gpt2;
    Layers<uint64_t> *lm_head;

    SecGPT2LMHead(TransformerConfig &config, int vocab_size = 50257) : Layers<uint64_t>("SecGPT2LMHead"), config(config)
    {
        gpt2 = new SecMixedGPT2(config);
        lm_head = new SecLinear<uint64_t>(config.hidden_size, vocab_size);
    }

    twopc_protocols::ShareTensor<uint64_t> forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters) override
    {
        twopc_protocols::ShareTensor<uint64_t> outputs(x.shape);
        outputs = (*this->gpt2)(x, parameters);
        return (*this->lm_head)(outputs);
    }
};

class SecGPT2LMHead64 : public Layers<uint64_t>
{
public:
    TransformerConfig config;
    Layers<uint64_t> *gpt2;
    Layers<uint64_t> *lm_head;

    SecGPT2LMHead64(TransformerConfig &config, int vocab_size = 50257) : Layers<uint64_t>("SecGPT2LMHead"), config(config)
    {
        gpt2 = new SecGPT2<uint64_t>(config);
        lm_head = new SecLinear<uint64_t>(config.hidden_size, vocab_size);
    }

    twopc_protocols::ShareTensor<uint64_t> forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters) override
    {
        twopc_protocols::ShareTensor<uint64_t> outputs(x.shape);
        outputs = (*this->gpt2)(x, parameters);
        return (*this->lm_head)(outputs);
    }
};

template <typename T>
void utils::masked_softmax(twopc_protocols::ShareTensor<T> &x, twopc_protocols::ShareTensor<T> &res, Parameters<T> &parameter, bool malicious)
{
    perf_stats::Scope perf_scope("twopc_protocols::masked_softmax_forward");
    always_assert(x.shape[x.shape.size() - 1] == x.shape[x.shape.size() - 2]);

    std::vector<uint32_t> sum_shape = x.shape;
    uint32_t dim_size = sum_shape.back();
    sum_shape.pop_back();
    uint32_t second_dim = sum_shape.back();
    uint32_t mask_size = dim_size * second_dim;
    uint32_t other_dim = x.size() / mask_size;

    // mask: use the first value to replace triu
    int index;
    for (int i = 0; i < other_dim; i++)
    {
        for (int j = 0; j < second_dim; j++)
        {
            for (int k = 0; k < dim_size; k++)
            {
                if (j < k)
                {
                    index = i * mask_size + j * dim_size;
                    x.share.data[index + k] = x.share.data[index];
                }
            }
        }
    }

    twopc_protocols::ShareTensor<T> x_max(sum_shape), delta(x.shape);
    uint32_t common_size = x_max.size();
    twopc_protocols::max_last_dim(x, x_max, parameter, malicious);

#pragma omp parallel for if (utils::should_parallelize(static_cast<uint64_t>(common_size) * dim_size))
    for (int i = 0; i < common_size; i++)
    {
        for (int j = 0; j < dim_size; j++)
        {
            index = i * dim_size + j;
            delta.share.data[index] = x_max.share.data[i] - x.share.data[index];
        }
    }

    twopc_protocols::ShareTensor<T> exp_x(x.shape);
    twopc_protocols::neg_exp(delta, exp_x, parameter, malicious);
    twopc_protocols::ShareTensor<T> sum(sum_shape);

    // only need to sum the tril
#pragma omp parallel for if (utils::should_parallelize(static_cast<uint64_t>(other_dim) * mask_size))
    for (int i = 0; i < other_dim; i++)
    {
        for (int j = 0; j < second_dim; j++)
        {
            index = i * second_dim + j;

            sum.share.data[index] = 0;

            for (int k = 0; k < dim_size; k++)
            {
                if (j >= k)
                {
                    sum.share.data[index] += exp_x.share.data[index * dim_size + k];
                }
            }
        }
    }

    twopc_protocols::inv(sum, sum, dim_size, parameter, malicious);
    twopc_protocols::ShareTensor<T> broadcast_sum(x.shape);
#pragma omp parallel for if (utils::should_parallelize(static_cast<uint64_t>(common_size) * dim_size))
    for (int i = 0; i < sum.size(); i++)
    {
        for (int j = 0; j < dim_size; j++)
        {
            broadcast_sum.share.data[i * dim_size + j] = sum.share.data[i];
        }
    }
    twopc_protocols::mul(exp_x, broadcast_sum, res, true, malicious);
    mask(res);
}

template <typename T>
twopc_protocols::ShareTensor<T> utils::img2col_pool(twopc_protocols::ShareTensor<T> &x, const int kernel_size, const int stride, const int pad)
{
    always_assert(x.shape.size() == 4);
    uint32_t batch = x.shape[0], channel = x.shape[1], height_origin = x.shape[2], width_origin = x.shape[3];

    const uint32_t height_padded = height_origin + 2 * pad;
    const uint32_t width_padded = width_origin + 2 * pad;

    const uint32_t height_out = (height_padded - kernel_size) / stride + 1;
    const uint32_t width_out = (width_padded - kernel_size) / stride + 1;

    const uint32_t out_size = height_out * width_out;
    const uint32_t k_square = kernel_size * kernel_size;
    twopc_protocols::ShareTensor<T> output({batch, channel, out_size, k_square});

    const uint32_t origin_shape = height_origin * width_origin;
    const uint32_t channel_origin_shape = channel * origin_shape;
    const uint32_t out_shape = out_size * k_square;
    const uint32_t channel_out_shape = channel * out_shape;

    for (int b = 0; b < batch; ++b)
    {
        const uint32_t b_offset = b * channel_origin_shape;
        const uint32_t b_out_offset = b * channel_out_shape;

        for (int kh = 0; kh < kernel_size; ++kh)
        {
            for (int kw = 0; kw < kernel_size; ++kw)
            {
                const uint32_t kernel_element = kh * kernel_size + kw;

                for (int c = 0; c < channel; ++c)
                {
                    const uint32_t c_input_offset = b_offset + c * origin_shape;
                    const uint32_t c_output_offset = b_out_offset + c * out_shape + kernel_element;

                    for (int h = 0; h < height_out; ++h)
                    {
                        const int h_start = h * stride - pad + kh;
                        const uint32_t h_out_base = h * width_out;
                        const bool h_valid = (h_start >= 0) && (h_start < height_origin);

                        for (int w = 0; w < width_out; ++w)
                        {
                            const int w_start = w * stride - pad + kw;

                            const uint32_t output_idx = c_output_offset + (h_out_base + w) * k_square;

                            if (h_valid && (w_start >= 0) && (w_start < width_origin))
                            {
                                const uint32_t input_idx = c_input_offset + h_start * width_origin + w_start;
                                output.share.data[output_idx] = x.share.data[input_idx];
                            }
                            else
                            {
                                output.share.data[output_idx] = T(0);
                            }
                        }
                    }
                }
            }
        }
    }

    return output;
}

template <typename T>
twopc_protocols::ShareTensor<T> utils::img2col_conv(twopc_protocols::ShareTensor<T> &x, const int kernel_size, const int stride, const int pad)
{
    always_assert(x.shape.size() == 4);
    const uint32_t batch = x.shape[0];
    const uint32_t channel = x.shape[1];
    const uint32_t height_origin = x.shape[2];
    const uint32_t width_origin = x.shape[3];

    const uint32_t height_padded = height_origin + 2 * pad;
    const uint32_t width_padded = width_origin + 2 * pad;

    const uint32_t height_out = (height_padded - kernel_size) / stride + 1;
    const uint32_t width_out = (width_padded - kernel_size) / stride + 1;

    const uint32_t out_size = height_out * width_out;
    const uint32_t k_square = kernel_size * kernel_size;

    twopc_protocols::ShareTensor<T> output({batch, out_size, channel * k_square});

    // precompute steps
    const uint32_t batch_stride = out_size * channel * k_square;
    const uint32_t out_dim_stride = channel * k_square;

#pragma omp parallel for collapse(3) schedule(static)
    for (int b = 0; b < batch; ++b)
    {
        for (int h = 0; h < height_out; ++h)
        {
            for (int w = 0; w < width_out; ++w)
            {
                const uint32_t pos = h * width_out + w;
                const uint32_t pos_offset = b * batch_stride + pos * out_dim_stride;

                for (int kh = 0; kh < kernel_size; ++kh)
                {
                    const int h_start = h * stride - pad + kh;
                    const bool h_valid = (h_start >= 0) && (h_start < height_origin);

                    for (int kw = 0; kw < kernel_size; ++kw)
                    {
                        const int w_start = w * stride - pad + kw;
                        const uint32_t kernel_element = kh * kernel_size + kw;

                        for (int c = 0; c < channel; ++c)
                        {
                            // [b][pos][c*k_square + kernel_element]
                            const uint32_t output_idx = pos_offset + c * k_square + kernel_element;

                            const uint32_t input_base = b * channel * height_origin * width_origin + c * height_origin * width_origin;

                            if (h_valid && (w_start >= 0) && (w_start < width_origin))
                            {
                                const uint32_t input_idx = input_base + h_start * width_origin + w_start;
                                output.share.data[output_idx] = x.share.data[input_idx];
                            }
                            else
                            {
                                output.share.data[output_idx] = T(0);
                            }
                        }
                    }
                }
            }
        }
    }

    return output;
}

#endif // LAYERS_H
