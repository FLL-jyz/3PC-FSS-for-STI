#include "party2pc.h"
#include "twopc_protocols.h"
#include "ass_protocols.h"
#include "test_base.h"
#include "llmConfig.h"
#include "globals.h"
#include <vector>
#include <string>
#include "fss.h"
#include "2pcdpf.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <random>
#include <omp.h>


static uint32_t batch_size = 1;
static bool print_res = false;
static int loop = 5;
static int warmup_rounds = 3;
static constexpr size_t BLOCK_BYTES = 16;
static bool Verification = false;
static constexpr int kOfflineOmpThreads = 8;

static constexpr size_t IKNP_SECURITY_PARAM = 128;
static constexpr size_t IKNP_BLOCK_BYTES = 16;
static constexpr size_t IKNP_BOOLEAN_COT_PER_TRIPLE = 2;
static constexpr size_t RING_WORD_BITS = 64;

static void simulate_offline_comm_bytes(size_t bytes);

struct ModelBeaverSummary
{
    std::string name;
    size_t layers;
    size_t seq_len;
    size_t hidden;
    size_t intermediate;
    size_t heads;
    size_t matmul_multiplications;
    size_t softmax_elementwise_multiplications;
    size_t norm_elementwise_multiplications;
    size_t activation_elementwise_multiplications;
    size_t total_multiplications;
};

enum class OfflineExecutionMode
{
    Local,
    Analysis,
    DealerBench,
    Emulate,
};

enum class OfflineWorkloadKind
{
    A2b,
    Drelu,
    Lut,
    SineMask,
};

struct OfflinePrimitiveWorkload
{
    OfflineWorkloadKind kind;
    std::string label;
    uint32_t bit_len;
    uint64_t count;
    std::string shape;
    uint64_t adder_invocations = 0;
};

struct ModelOfflineProfile
{
    std::string name;
    TransformerConfig config;
    std::vector<OfflinePrimitiveWorkload> workloads;
    std::vector<OfflinePrimitiveWorkload> precise_a2b_calls;
    ModelBeaverSummary beaver_summary;
};

struct DealerDistributionSummary
{
    uint64_t per_party_exact_bytes = 0;
    uint64_t total_exact_dealer_bytes = 0;
    uint64_t per_party_upper_bound_bytes = 0;
    uint64_t total_upper_bound_dealer_bytes = 0;
    uint64_t rounds_per_party = 0;
    uint64_t total_send_actions = 0;
    uint64_t total_a2b = 0;
    uint64_t total_sine_mask = 0;
    uint64_t total_a2b_effective_bits = 0;
    uint64_t total_a2b_impl_boolean_triples = 0;
    uint64_t total_a2b_adder_invocations = 0;
};

struct BeaverDistributionSummary
{
    uint64_t per_party_bytes = 0;
    uint64_t total_dealer_bytes = 0;
    uint64_t rounds_per_party = 0;
    uint64_t total_send_actions = 0;
    uint64_t total_triples = 0;
    double estimated_seconds = 0.0;
};

static constexpr uint32_t kOfflineMeasuredLayers = 1;

static uint32_t offline_measured_layers_override()
{
    const char *value = std::getenv("SHAFT_OFFLINE_LLM_MEASURED_LAYERS");
    if (value == nullptr)
    {
        return 0;
    }

    const long parsed = std::strtol(value, nullptr, 10);
    return parsed > 0 ? static_cast<uint32_t>(parsed) : 0;
}

static uint32_t offline_seq_len_override()
{
    const char *value = std::getenv("SHAFT_OFFLINE_LLM_SEQ_LEN");
    if (value == nullptr)
    {
        return 0;
    }

    const long parsed = std::strtol(value, nullptr, 10);
    return parsed > 0 ? static_cast<uint32_t>(parsed) : 0;
}

static uint32_t resolved_offline_seq_len(uint32_t default_seq_len)
{
    const uint32_t override_seq_len = offline_seq_len_override();
    return override_seq_len == 0 ? default_seq_len : override_seq_len;
}

struct IKNPPerfSample
{
    size_t sampled_ots = 0;
    double seconds = 0.0;
    double ot_per_sec = 0.0;
};

struct A2BAdderPerfSample
{
    uint64_t sampled_instances = 0;
    uint32_t adder_bit_len = 0;
    uint64_t sampled_boolean_triples = 0;
    double seconds = 0.0;
    double instances_per_sec = 0.0;
    double boolean_triples_per_sec = 0.0;
};

struct IKNPBeaverPerfSample
{
    size_t sampled_triples = 0;
    double seconds = 0.0;
    double triples_per_sec = 0.0;
};

struct DealerBeaverPerfSample
{
    size_t sampled_triples = 0;
    double seconds = 0.0;
    double triples_per_sec = 0.0;
};

struct OfflineMaterialPerfSample
{
    uint64_t sampled_instances = 0;
    double seconds = 0.0;
    double instances_per_sec = 0.0;
};

struct BoolBeaverShares
{
    uint8_t a1;
    uint8_t b1;
    uint8_t c1;
    uint8_t a2;
    uint8_t b2;
    uint8_t c2;
};

static IKNPPerfSample benchmark_iknp_extension(size_t sampled_ots);
static A2BAdderPerfSample benchmark_a2b_boolean_adder_material(uint32_t bit_len, uint64_t sampled_instances);
static OfflineMaterialPerfSample benchmark_sine_mask_generation(uint64_t sampled_instances);

class IKNPExtensionFactory
{
public:
    explicit IKNPExtensionFactory(size_t base_ot = IKNP_SECURITY_PARAM)
        : base_ot_count_(base_ot), seed_bytes_(base_ot * IKNP_BLOCK_BYTES)
    {
    }

    size_t base_ot_count() const { return base_ot_count_; }
    size_t seed_bytes() const { return seed_bytes_; }

    void extend_ots(size_t num_ots)
    {
        // Simulate IKNP extension in the same spirit as: PRG expand + transpose.
        (void)benchmark_iknp_extension(num_ots);
    }

private:
    size_t base_ot_count_;
    size_t seed_bytes_;
};

static inline void compute_cross_term_cot(uint8_t sender_a, uint8_t receiver_b, std::mt19937_64 &rng,
                                          uint8_t &sender_share, uint8_t &receiver_share)
{
    // COT message pattern: m0 = r, m1 = r XOR a
    sender_share = static_cast<uint8_t>(rng() & 1ULL);
    const uint8_t m0 = sender_share;
    const uint8_t m1 = static_cast<uint8_t>(sender_share ^ sender_a);
    receiver_share = receiver_b ? m1 : m0;
}

static BoolBeaverShares generate_boolean_beaver_triple_iknp(std::mt19937_64 &rng)
{
    BoolBeaverShares out;
    out.a1 = static_cast<uint8_t>(rng() & 1ULL);
    out.b1 = static_cast<uint8_t>(rng() & 1ULL);
    out.a2 = static_cast<uint8_t>(rng() & 1ULL);
    out.b2 = static_cast<uint8_t>(rng() & 1ULL);

    uint8_t s_a_for_a1b2 = 0;
    uint8_t s_b_for_a1b2 = 0;
    uint8_t s_b_for_a2b1 = 0;
    uint8_t s_a_for_a2b1 = 0;

    // Cross term 1: a1 & b2 (Alice sender, Bob receiver)
    compute_cross_term_cot(out.a1, out.b2, rng, s_a_for_a1b2, s_b_for_a1b2);
    // Cross term 2: a2 & b1 (Bob sender, Alice receiver)
    compute_cross_term_cot(out.a2, out.b1, rng, s_b_for_a2b1, s_a_for_a2b1);

    // Assemble Beaver c-shares under XOR sharing.
    out.c1 = static_cast<uint8_t>((out.a1 & out.b1) ^ s_a_for_a1b2 ^ s_a_for_a2b1);
    out.c2 = static_cast<uint8_t>((out.a2 & out.b2) ^ s_b_for_a1b2 ^ s_b_for_a2b1);
    return out;
}

static inline bool mul_overflow_u64(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr)
    {
        return true;
    }
    if (a == 0 || b == 0)
    {
        *out = 0;
        return false;
    }
    if (a > std::numeric_limits<uint64_t>::max() / b)
    {
        return true;
    }
    *out = a * b;
    return false;
}

static inline bool add_overflow_u64(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr)
    {
        return true;
    }
    if (b > std::numeric_limits<uint64_t>::max() - a)
    {
        return true;
    }
    *out = a + b;
    return false;
}

static uint32_t lut_index_bitlength(uint64_t table_size)
{
    uint32_t bit_len = 0;
    uint64_t size = std::max<uint64_t>(table_size, 1);
    while ((1ULL << bit_len) < size)
    {
        ++bit_len;
    }
    return bit_len;
}

static uint32_t default_pc_cmp_bitlength()
{
    return static_cast<uint32_t>(std::log2(2.0 * sizeof(uint64_t) * 8.0) + 1.0);
}

static uint32_t max_last_dim_eval_bitlength()
{
    return 52;
}

static uint64_t a2b_adder_and_gates(uint32_t bit_len)
{
    if (bit_len == 0)
    {
        return 0;
    }
    return 2ULL * bit_len - 1ULL;
}

static uint32_t actual_a2b_adder_bitlength()
{
    return 64;
}

static uint32_t effective_a2b_adder_bitlength(uint32_t eval_bit_len)
{
    return eval_bit_len == 0 ? actual_a2b_adder_bitlength() : eval_bit_len;
}

static uint32_t runtime_a2b_adder_bitlength(uint32_t eval_bit_len)
{
    return effective_a2b_adder_bitlength(eval_bit_len);
}

static uint32_t inv_chopped_lut_bitlength(uint32_t max_k)
{
    uint32_t int_bits = 0;
    while ((1U << int_bits) < (max_k + 1U))
    {
        ++int_bits;
    }
    return 6U + int_bits;
}

static uint32_t rsqrt_getk_compare_fanout()
{
    return static_cast<uint32_t>(std::log(std::pow(2.0, 2.0 * FLOAT_PRECISION_64)) / std::log(static_cast<double>(SCALE_BASE)));
}

static void append_workload(std::vector<OfflinePrimitiveWorkload> &workloads,
                            OfflineWorkloadKind kind,
                            const std::string &label,
                            uint32_t bit_len,
                            uint64_t count)
{
    if (count == 0)
    {
        return;
    }
    workloads.push_back({kind, label, bit_len, count, std::string(), 0});
}

static void append_precise_workload(std::vector<OfflinePrimitiveWorkload> &workloads,
                                    OfflineWorkloadKind kind,
                                    const std::string &label,
                                    uint32_t bit_len,
                                    uint64_t count,
                                    const std::string &shape,
                                    uint64_t adder_invocations)
{
    if (count == 0)
    {
        return;
    }
    workloads.push_back({kind, label, bit_len, count, shape, adder_invocations});
}

static const char *offline_workload_name(OfflineWorkloadKind kind)
{
    switch (kind)
    {
    case OfflineWorkloadKind::A2b:
        return "offline_a2b";
    case OfflineWorkloadKind::Drelu:
        return "offline_drelu";
    case OfflineWorkloadKind::Lut:
        return "offline_lut";
    case OfflineWorkloadKind::SineMask:
        return "offline_sine_mask";
    }
    return "offline_unknown";
}

static const char *offline_workload_kind_name(OfflineWorkloadKind kind)
{
    switch (kind)
    {
    case OfflineWorkloadKind::A2b:
        return "a2b";
    case OfflineWorkloadKind::Drelu:
        return "drelu";
    case OfflineWorkloadKind::Lut:
        return "lut";
    case OfflineWorkloadKind::SineMask:
        return "sine-mask";
    }
    return "unknown";
}

static bool is_legacy_offline_workload_kind(OfflineWorkloadKind kind)
{
    return kind == OfflineWorkloadKind::Drelu || kind == OfflineWorkloadKind::Lut;
}

static uint64_t ceil_div_u64(uint64_t numerator, uint64_t denominator)
{
    if (denominator == 0)
    {
        return 0;
    }
    return (numerator + denominator - 1) / denominator;
}

static std::string shape_to_string(std::initializer_list<uint64_t> dims)
{
    std::string result = "[";
    bool first = true;
    for (const uint64_t dim : dims)
    {
        if (!first)
        {
            result += "x";
        }
        result += std::to_string(dim);
        first = false;
    }
    result += "]";
    return result;
}

static uint64_t sum_max_last_dim_select_instances(uint64_t freeze_size, uint64_t last_dim_size)
{
    uint64_t total = 0;
    uint64_t current = last_dim_size;
    while (current > 1)
    {
        const uint64_t half = (current + 1) / 2;
        total += freeze_size * half;
        current = half;
    }
    return total;
}

static void append_layerwise_a2b_calls(std::vector<OfflinePrimitiveWorkload> &workloads,
                                       const std::string &base_label,
                                       uint32_t bit_len,
                                       uint64_t layers,
                                       uint64_t instance_count,
                                       const std::string &shape)
{
    for (uint64_t layer = 0; layer < layers; ++layer)
    {
        append_precise_workload(workloads,
                                OfflineWorkloadKind::A2b,
                                base_label + ".layer" + std::to_string(layer + 1),
                                bit_len,
                                instance_count,
                                shape,
                                1);
    }
}

static void append_softmax_max_last_dim_calls(std::vector<OfflinePrimitiveWorkload> &workloads,
                                              uint64_t layers,
                                              uint64_t batch,
                                              uint64_t heads,
                                              uint64_t seq_len)
{
    const uint64_t freeze_size = batch * heads * seq_len;
    for (uint64_t layer = 0; layer < layers; ++layer)
    {
        uint64_t current_last_dim = seq_len;
        uint64_t round = 0;
        while (current_last_dim > 1)
        {
            const uint64_t half = (current_last_dim + 1) / 2;
            append_precise_workload(workloads,
                                    OfflineWorkloadKind::A2b,
                                    "softmax.max_last_dim.select.layer" + std::to_string(layer + 1) +
                                        ".round" + std::to_string(round + 1) +
                                        "(" + std::to_string(current_last_dim) + "->" + std::to_string(half) + ")",
                                    max_last_dim_eval_bitlength(),
                                    freeze_size * half,
                                    shape_to_string({batch, heads, seq_len, half}),
                                    1);
            current_last_dim = half;
            ++round;
        }
    }
}

static ModelBeaverSummary build_model_summary(const std::string &model_name, const TransformerConfig &config)
{
    const uint64_t batch = batch_size;
    const uint64_t seq_len = config.seq_len;
    const uint64_t hidden = config.hidden_size;
    const uint64_t intermediate = config.intermediate_size;
    const uint64_t heads = config.num_attention_heads;
    const uint64_t layers = config.num_layers;
    const uint64_t head_dim = hidden / heads;
    const uint64_t tokens = batch * seq_len;
    const uint64_t bh = batch * heads;
    const bool is_llama = (model_name.find("llama2") != std::string::npos);

    uint64_t per_layer_matmul = 0;
    uint64_t term = 0;

    // MHA qkv projection
    term = tokens * hidden * (3 * hidden);
    per_layer_matmul += term;
    // MHA qk^T
    term = (bh * seq_len) * head_dim * seq_len;
    per_layer_matmul += term;
    // MHA attn*v
    term = (bh * seq_len) * seq_len * head_dim;
    per_layer_matmul += term;
    // MHA output projection
    term = tokens * hidden * hidden;
    per_layer_matmul += term;

    // FFN projections:
    // - BERT/GPT2: up + down
    // - Llama2: gate + up + down
    term = tokens * hidden * intermediate;
    per_layer_matmul += term;
    if (is_llama)
    {
        term = tokens * hidden * intermediate;
        per_layer_matmul += term;
    }
    term = tokens * intermediate * hidden;
    per_layer_matmul += term;

    const uint64_t matmul_total = per_layer_matmul * layers;

    // Softmax elementwise multiplications in protocol (approx):
    // 1) neg_exp square, 2) neg_exp final multiply, 3) softmax normalize multiply
    const uint64_t softmax_sites = layers * batch * heads * seq_len * seq_len;
    const uint64_t softmax_elem_mul = softmax_sites * 3;

    // Norm elementwise multiplications in protocol (approx):
    // 1) square accumulation, 2) normalize, 3) affine scale
    const uint64_t norm_sites = (2 * layers + 1) * batch * seq_len * hidden;
    const uint64_t norm_elem_mul = norm_sites * 3;

    // Activation elementwise multiplications:
    // - BERT/GPT2 GELU: ~1 mul per activation output
    // - Llama2 SwiGLU: final gate*up multiply per activation output
    const uint64_t act_sites = layers * batch * seq_len * intermediate;
    const uint64_t act_elem_mul = act_sites;

    uint64_t total_mul = matmul_total;
    total_mul += softmax_elem_mul;
    total_mul += norm_elem_mul;
    total_mul += act_elem_mul;

    ModelBeaverSummary summary;
    summary.name = model_name;
    summary.layers = layers;
    summary.seq_len = seq_len;
    summary.hidden = hidden;
    summary.intermediate = intermediate;
    summary.heads = heads;
    summary.matmul_multiplications = matmul_total;
    summary.softmax_elementwise_multiplications = softmax_elem_mul;
    summary.norm_elementwise_multiplications = norm_elem_mul;
    summary.activation_elementwise_multiplications = act_elem_mul;
    summary.total_multiplications = total_mul;
    return summary;
}

static ModelOfflineProfile build_model_offline_profile(const std::string &model_name, const TransformerConfig &config)
{
    const bool is_llama = (model_name.find("llama2") != std::string::npos);
    const uint64_t batch = batch_size;
    const uint64_t layers = config.num_layers;
    const uint64_t seq_len = config.seq_len;
    const uint64_t heads = config.num_attention_heads;
    const uint64_t intermediate = config.intermediate_size;
    const uint64_t tokens = batch * seq_len;
    const uint64_t attention_sites_per_layer = batch * heads * seq_len * seq_len;
    const uint64_t max_reduction_sites_per_layer = sum_max_last_dim_select_instances(batch * heads * seq_len, seq_len);
    const std::string activation_shape = shape_to_string({batch, seq_len, intermediate});
    const std::string softmax_shape = shape_to_string({batch, heads, seq_len, seq_len});

    const uint64_t activation_count = layers * tokens * intermediate;
    const uint64_t softmax_sites = layers * attention_sites_per_layer;
    const uint64_t softmax_max_count = layers * max_reduction_sites_per_layer;

    ModelOfflineProfile profile{model_name, config, {}, {}, build_model_summary(model_name, config)};

    if (!is_llama)
    {
        append_workload(profile.workloads, OfflineWorkloadKind::A2b, "gelu.select_relu", 48, activation_count);
        append_workload(profile.workloads, OfflineWorkloadKind::A2b, "gelu.in_range", 48, activation_count);
        append_workload(profile.workloads, OfflineWorkloadKind::SineMask, "gelu.secure_sin_mask", 64, activation_count * 8ULL);

        append_layerwise_a2b_calls(profile.precise_a2b_calls,
                                   "gelu.select_relu",
                                   48,
                                   layers,
                                   tokens * intermediate,
                                   activation_shape);
        append_layerwise_a2b_calls(profile.precise_a2b_calls,
                                   "gelu.in_range",
                                   48,
                                   layers,
                                   tokens * intermediate,
                                   activation_shape);
    }

    append_workload(profile.workloads, OfflineWorkloadKind::A2b, "softmax.max_last_dim.select", max_last_dim_eval_bitlength(), softmax_max_count);
    append_workload(profile.workloads, OfflineWorkloadKind::A2b, "softmax.clip_lower", 52, softmax_sites);
    append_workload(profile.workloads, OfflineWorkloadKind::A2b, "softmax.clip_upper", 52, softmax_sites);

    append_softmax_max_last_dim_calls(profile.precise_a2b_calls, layers, batch, heads, seq_len);
    append_layerwise_a2b_calls(profile.precise_a2b_calls,
                               "softmax.clip_lower",
                               52,
                               layers,
                               attention_sites_per_layer,
                               softmax_shape);
    append_layerwise_a2b_calls(profile.precise_a2b_calls,
                               "softmax.clip_upper",
                               52,
                               layers,
                               attention_sites_per_layer,
                               softmax_shape);

    return profile;
}

static TransformerConfig make_measured_offline_config(const TransformerConfig &config)
{
    TransformerConfig measured = config;
    const uint32_t override_layers = offline_measured_layers_override();
    const uint32_t measured_layers = override_layers == 0
                                         ? kOfflineMeasuredLayers
                                         : std::min<uint32_t>(override_layers, config.num_layers);
    measured.num_layers = std::min<uint32_t>(config.num_layers, measured_layers);
    return measured;
}

static IKNPPerfSample benchmark_iknp_extension(size_t sampled_ots)
{
    omp_set_num_threads(kOfflineOmpThreads);
    // IKNP transpose benchmark: generate a k x n pseudo-random bit matrix, then transpose to n x k.
    const size_t k = IKNP_SECURITY_PARAM;
    const size_t n = std::max<size_t>(sampled_ots, 1024);
    const size_t words_per_row = (n + 63) / 64;
    const size_t words_per_col = (k + 63) / 64;

    std::vector<uint64_t> q(k * words_per_row, 0);
    std::vector<uint64_t> qt(n * words_per_col, 0);

    std::mt19937_64 seed_rng(0xC0FFEEULL);
    std::vector<uint64_t> seeds(k, 0);
    for (size_t i = 0; i < k; ++i)
    {
        seeds[i] = seed_rng();
    }

    const auto begin = std::chrono::steady_clock::now();

    #pragma omp parallel for
    for (long long row = 0; row < static_cast<long long>(k); ++row)
    {
        uint64_t st = seeds[row] ^ 0x9e3779b97f4a7c15ULL;
        for (size_t w = 0; w < words_per_row; ++w)
        {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            q[row * words_per_row + w] = st;
        }
    }

    #pragma omp parallel for
    for (long long col = 0; col < static_cast<long long>(n); ++col)
    {
        const size_t col_index = static_cast<size_t>(col);
        for (size_t row = 0; row < k; ++row)
        {
            const size_t row_base = row * words_per_row;
            const uint64_t word = q[row_base + (col_index >> 6)];
            const uint64_t bit = (word >> (col_index & 63)) & 1ULL;
            if (bit)
            {
                qt[col_index * words_per_col + (row >> 6)] |= (1ULL << (row & 63));
            }
        }
    }

    const auto end = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(end - begin).count();
    IKNPPerfSample perf;
    perf.sampled_ots = n;
    perf.seconds = sec;
    perf.ot_per_sec = sec > 0.0 ? static_cast<double>(n) / sec : 0.0;
    return perf;
}

static IKNPBeaverPerfSample benchmark_iknp_beaver_generation(size_t sampled_triples)
{
    omp_set_num_threads(kOfflineOmpThreads);
    const size_t n = std::max<size_t>(sampled_triples, 1024);
    IKNPExtensionFactory iknp;
    iknp.extend_ots(n * IKNP_BOOLEAN_COT_PER_TRIPLE);

    uint64_t checksum = 0;
    const auto begin = std::chrono::steady_clock::now();

    #pragma omp parallel
    {
        std::mt19937_64 rng(0xBEEF1234ULL + static_cast<uint64_t>(omp_get_thread_num()));
        uint64_t thread_checksum = 0;

        #pragma omp for nowait
        for (long long i = 0; i < static_cast<long long>(n); ++i)
        {
            const BoolBeaverShares tri = generate_boolean_beaver_triple_iknp(rng);
            thread_checksum += static_cast<uint64_t>(tri.c1 ^ tri.c2);
        }

        #pragma omp atomic
        checksum += thread_checksum;
    }

    const auto end = std::chrono::steady_clock::now();
    (void)checksum;
    const double sec = std::chrono::duration<double>(end - begin).count();

    IKNPBeaverPerfSample perf;
    perf.sampled_triples = n;
    perf.seconds = sec;
    perf.triples_per_sec = sec > 0.0 ? static_cast<double>(n) / sec : 0.0;
    return perf;
}

static double benchmark_random_material_generation_bytes(size_t bytes, uint64_t seed)
{
    omp_set_num_threads(kOfflineOmpThreads);
    if (bytes == 0)
    {
        return 0.0;
    }

    const size_t total_words = (bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    uint64_t checksum = 0;
    const auto begin = std::chrono::steady_clock::now();

    #pragma omp parallel
    {
        std::mt19937_64 rng(seed + static_cast<uint64_t>(omp_get_thread_num()));
        uint64_t thread_checksum = 0;

        #pragma omp for nowait
        for (long long index = 0; index < static_cast<long long>(total_words); ++index)
        {
            const uint64_t value = rng();
            thread_checksum ^= value + static_cast<uint64_t>(index);
        }

        #pragma omp critical
        {
            checksum ^= thread_checksum;
        }
    }

    const auto end = std::chrono::steady_clock::now();
    (void)checksum;
    return std::chrono::duration<double>(end - begin).count();
}

static A2BAdderPerfSample benchmark_a2b_boolean_adder_material(uint32_t bit_len, uint64_t sampled_instances)
{
    const uint64_t sample_count = std::max<uint64_t>(sampled_instances, 1024ULL);
    const uint32_t effective_bit_len = effective_a2b_adder_bitlength(bit_len);
    const uint64_t sampled_boolean_triples = sample_count * a2b_adder_and_gates(effective_bit_len);

    omp_set_num_threads(kOfflineOmpThreads);
    uint64_t checksum = 0;
    const auto begin = std::chrono::steady_clock::now();

    #pragma omp parallel
    {
        std::mt19937_64 rng(0xB00A2BULL + static_cast<uint64_t>(omp_get_thread_num()));
        uint64_t thread_checksum = 0;

        #pragma omp for nowait
        for (long long triple_index = 0; triple_index < static_cast<long long>(sampled_boolean_triples); ++triple_index)
        {
            const uint8_t u = static_cast<uint8_t>(rng() & 1ULL);
            const uint8_t v = static_cast<uint8_t>(rng() & 1ULL);
            const uint8_t w = static_cast<uint8_t>(u & v);
            const uint8_t u0 = static_cast<uint8_t>(rng() & 1ULL);
            const uint8_t v0 = static_cast<uint8_t>(rng() & 1ULL);
            const uint8_t w0 = static_cast<uint8_t>(rng() & 1ULL);
            const uint8_t u1 = static_cast<uint8_t>(u ^ u0);
            const uint8_t v1 = static_cast<uint8_t>(v ^ v0);
            const uint8_t w1 = static_cast<uint8_t>(w ^ w0);

            thread_checksum ^= static_cast<uint64_t>(u0 | (v0 << 1) | (w0 << 2) | (u1 << 3) | (v1 << 4) | (w1 << 5));
            thread_checksum ^= static_cast<uint64_t>(triple_index + 1);
        }

        #pragma omp critical
        {
            checksum ^= thread_checksum;
        }
    }

    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - begin).count();
    (void)checksum;

    A2BAdderPerfSample perf;
    perf.sampled_instances = sample_count;
    perf.adder_bit_len = effective_bit_len;
    perf.sampled_boolean_triples = sampled_boolean_triples;
    perf.seconds = seconds;
    perf.instances_per_sec = seconds > 0.0 ? static_cast<double>(sample_count) / seconds : 0.0;
    perf.boolean_triples_per_sec = seconds > 0.0 ? static_cast<double>(sampled_boolean_triples) / seconds : 0.0;
    return perf;
}

static OfflineMaterialPerfSample benchmark_sine_mask_generation(uint64_t sampled_instances)
{
    const uint64_t sample_count = std::max<uint64_t>(sampled_instances, 1024ULL);
    const uint64_t share_words = 3ULL * sample_count;
    const uint64_t bytes = share_words * sizeof(uint64_t);
    const double seconds = benchmark_random_material_generation_bytes(static_cast<size_t>(bytes), 0x51A3B5C7ULL);

    OfflineMaterialPerfSample sample;
    sample.sampled_instances = sample_count;
    sample.seconds = seconds;
    sample.instances_per_sec = seconds > 0.0 ? static_cast<double>(sample_count) / seconds : 0.0;
    return sample;
}

static DealerBeaverPerfSample benchmark_third_party_dealer_generation(size_t sampled_triples)
{
    omp_set_num_threads(kOfflineOmpThreads);
    const size_t n = std::max<size_t>(sampled_triples, 1024);
    uint64_t checksum = 0;

    const auto begin = std::chrono::steady_clock::now();
    #pragma omp parallel
    {
        std::mt19937_64 rng(0x12345678ULL + static_cast<uint64_t>(omp_get_thread_num()));
        uint64_t thread_checksum = 0;

        #pragma omp for nowait
        for (long long i = 0; i < static_cast<long long>(n); ++i)
        {
            const uint64_t a = rng();
            const uint64_t b = rng();
            const uint64_t c = a * b;
            thread_checksum ^= c;
        }

        #pragma omp critical
        {
            checksum ^= thread_checksum;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    (void)checksum;

    DealerBeaverPerfSample perf;
    perf.sampled_triples = n;
    perf.seconds = std::chrono::duration<double>(end - begin).count();
    perf.triples_per_sec = perf.seconds > 0.0 ? static_cast<double>(n) / perf.seconds : 0.0;
    return perf;
}

static void print_model_dealer_report_if_p0(const ModelBeaverSummary &summary, const DealerBeaverPerfSample &perf)
{
    auto &party = Party2PC::getInstance();
    if (party.party_id != 0)
    {
        return;
    }

    const uint64_t triples = summary.total_multiplications;
    const double est_seconds = perf.triples_per_sec > 0.0
                                   ? static_cast<double>(triples) / perf.triples_per_sec
                                   : 0.0;

    // If dealer uses PRG to derive (a,b), minimal seed state can be 2 streams.
    const size_t dealer_prg_seed_bytes = 2 * IKNP_BLOCK_BYTES;

    std::cout << "[Dealer Beaver Offline] Model=" << summary.name
              << " (layers=" << summary.layers
              << ", seq=" << summary.seq_len
              << ", hidden=" << summary.hidden
              << ", intermediate=" << summary.intermediate
              << ", heads=" << summary.heads << ")" << std::endl;
    std::cout << "  matmul multiplications=" << summary.matmul_multiplications << std::endl;
    std::cout << "  softmax elementwise multiplications=" << summary.softmax_elementwise_multiplications << std::endl;
    std::cout << "  LN/RMSNorm elementwise multiplications=" << summary.norm_elementwise_multiplications << std::endl;
    std::cout << "  GELU/SwiGLU elementwise multiplications=" << summary.activation_elementwise_multiplications << std::endl;
    std::cout << "  total multiplications=" << summary.total_multiplications << std::endl;
    std::cout << "  required Beaver triples=" << triples << std::endl;
    std::cout << "  COT required=0 (trusted third-party dealer mode)" << std::endl;
    std::cout << "  dealer PRG seed storage(approx)=" << convert_bytes_to_string(dealer_prg_seed_bytes)
              << " (two 128-bit streams for a,b)" << std::endl;
    std::cout << "  dealer benchmark: sampled " << perf.sampled_triples
              << " triples in " << std::fixed << std::setprecision(6) << perf.seconds
              << " s, throughput=" << std::setprecision(2) << perf.triples_per_sec << " triples/s"
              << " (OpenMP threads=" << kOfflineOmpThreads << ")" << std::endl;
    std::cout << "  estimated offline dealer time=" << std::setprecision(3) << est_seconds << " s" << std::endl;
}

static void report_llm_dealer_beaver_offline_if_p0()
{
    const DealerBeaverPerfSample perf = benchmark_third_party_dealer_generation(1 << 20);

    const TransformerConfig bert("bert-base", 128);
    const TransformerConfig gpt2("gpt2", 64);
    const TransformerConfig llama2("llama2-7b", 16);

    std::vector<ModelBeaverSummary> models;
    models.push_back(build_model_summary("bert-base", bert));
    models.push_back(build_model_summary("gpt2", gpt2));
    models.push_back(build_model_summary("llama2-7b", llama2));

    for (const auto &m : models)
    {
        print_model_dealer_report_if_p0(m, perf);
    }
}

static void print_model_protocol_report_if_p0(const ModelOfflineProfile &profile, const DealerBeaverPerfSample &perf)
{
    auto &party = Party2PC::getInstance();
    if (party.party_id != 0)
    {
        return;
    }

    uint64_t total_a2b = 0;
    uint64_t total_sine_mask = 0;
    std::cout << "[Offline Protocol Report] Model=" << profile.name
              << " (layers=" << profile.config.num_layers
              << ", seq=" << profile.config.seq_len
              << ", hidden=" << profile.config.hidden_size
              << ", intermediate=" << profile.config.intermediate_size
              << ", heads=" << profile.config.num_attention_heads << ")" << std::endl;
    for (const auto &workload : profile.workloads)
    {
        std::cout << "  " << offline_workload_name(workload.kind)
                  << "(" << workload.bit_len << ")"
                  << "  label=" << workload.label
                  << ", count=" << workload.count << std::endl;
        if (workload.kind == OfflineWorkloadKind::A2b)
        {
            total_a2b += workload.count;
        }
        else if (workload.kind == OfflineWorkloadKind::SineMask)
        {
            total_sine_mask += workload.count;
        }
    }
    std::cout << "  total offline_a2b count=" << total_a2b << std::endl;
    std::cout << "  total offline_sine_mask count=" << total_sine_mask << std::endl;
    print_model_dealer_report_if_p0(profile.beaver_summary, perf);
}

static void report_llm_protocol_and_beaver_offline_if_p0()
{
    const DealerBeaverPerfSample perf = benchmark_third_party_dealer_generation(1 << 20);

    const TransformerConfig bert("bert-base", 128);
    const TransformerConfig gpt2("gpt2", 64);
    const TransformerConfig llama16("llama2-7b", 16);
    const TransformerConfig llama128("llama2-7b", 128);

    std::vector<ModelOfflineProfile> profiles;
    profiles.push_back(build_model_offline_profile("bert-base", bert));
    profiles.push_back(build_model_offline_profile("gpt2", gpt2));
    profiles.push_back(build_model_offline_profile("llama2-7b-16tok", llama16));
    profiles.push_back(build_model_offline_profile("llama2-7b-128tok", llama128));

    for (const auto &profile : profiles)
    {
        print_model_protocol_report_if_p0(profile, perf);
    }
}

static void print_model_iknp_report_if_p0(const ModelBeaverSummary &summary,
                                          const IKNPPerfSample &ot_perf,
                                          const IKNPBeaverPerfSample &beaver_perf)
{
    auto &party = Party2PC::getInstance();
    if (party.party_id != 0)
    {
        return;
    }

    const uint64_t triples = summary.total_multiplications;
    uint64_t cot_bool = 0;
    uint64_t cot_ring64 = 0;
    uint64_t tmp = 0;

    if (mul_overflow_u64(triples, IKNP_BOOLEAN_COT_PER_TRIPLE, &cot_bool))
    {
        cot_bool = std::numeric_limits<uint64_t>::max();
    }
    if (mul_overflow_u64(cot_bool, RING_WORD_BITS, &cot_ring64))
    {
        cot_ring64 = std::numeric_limits<uint64_t>::max();
    }

    const uint64_t sender_seed_bytes = 2ULL * IKNP_SECURITY_PARAM * IKNP_BLOCK_BYTES;
    const uint64_t receiver_seed_bytes = IKNP_SECURITY_PARAM * IKNP_BLOCK_BYTES;
    const uint64_t receiver_choice_bytes = (IKNP_SECURITY_PARAM + 7) / 8;
    uint64_t seed_total_bytes = 0;
    (void)add_overflow_u64(sender_seed_bytes, receiver_seed_bytes + receiver_choice_bytes, &seed_total_bytes);

    const long double iknp_matrix_bytes =
        static_cast<long double>(cot_ring64) * static_cast<long double>(IKNP_SECURITY_PARAM) / 8.0L;
    const double estimated_seconds =
        ot_perf.ot_per_sec > 0.0 ? static_cast<double>(static_cast<long double>(cot_ring64) / ot_perf.ot_per_sec) : 0.0;
    const double estimated_boolean_beaver_seconds =
        beaver_perf.triples_per_sec > 0.0 ? static_cast<double>(triples / beaver_perf.triples_per_sec) : 0.0;

    std::cout << "[IKNP Beaver Offline] Model=" << summary.name
              << " (layers=" << summary.layers
              << ", seq=" << summary.seq_len
              << ", hidden=" << summary.hidden
              << ", intermediate=" << summary.intermediate
              << ", heads=" << summary.heads << ")" << std::endl;
    std::cout << "  triples(total multiplications)=" << triples << std::endl;
    std::cout << "  COT count (boolean, 2 per triple)=" << cot_bool << std::endl;
    std::cout << "  COT count (64-bit ring via bit decomposition)=" << cot_ring64 << std::endl;
    std::cout << "  IKNP base-OT PRG seeds:" << std::endl;
    std::cout << "    sender seeds=" << convert_bytes_to_string(sender_seed_bytes)
              << ", receiver seeds=" << convert_bytes_to_string(receiver_seed_bytes)
              << ", receiver choice bits=" << convert_bytes_to_string(receiver_choice_bytes)
              << ", total stored=" << convert_bytes_to_string(seed_total_bytes) << std::endl;
    std::cout << "  IKNP extension communication (u-matrix, approx)="
              << convert_bytes_to_string(static_cast<size_t>(iknp_matrix_bytes)) << std::endl;
    std::cout << "  IKNP benchmark: sampled " << ot_perf.sampled_ots
              << " OTs in " << std::fixed << std::setprecision(6) << ot_perf.seconds
              << " s, throughput=" << std::setprecision(2) << ot_perf.ot_per_sec << " OT/s" << std::endl;
    std::cout << "  Beaver(COT) benchmark: sampled " << beaver_perf.sampled_triples
              << " triples in " << std::setprecision(6) << beaver_perf.seconds
              << " s, throughput=" << std::setprecision(2) << beaver_perf.triples_per_sec << " triples/s" << std::endl;
    std::cout << "  Estimated offline IKNP time for this model="
              << std::setprecision(3) << estimated_seconds << " s" << std::endl;
    std::cout << "  Estimated offline boolean-Beaver time for this model="
              << std::setprecision(3) << estimated_boolean_beaver_seconds << " s" << std::endl;
}

static void report_llm_iknp_beaver_offline_if_p0()
{
    const IKNPPerfSample ot_perf = benchmark_iknp_extension(1 << 18);
    const IKNPBeaverPerfSample beaver_perf = benchmark_iknp_beaver_generation(1 << 17);

    const TransformerConfig bert("bert-base", 128);
    const TransformerConfig gpt2("gpt2", 64);
    const TransformerConfig llama2("llama2-7b", 16);

    std::vector<ModelBeaverSummary> models;
    models.push_back(build_model_summary("bert-base", bert));
    models.push_back(build_model_summary("gpt2", gpt2));
    models.push_back(build_model_summary("llama2-7b", llama2));

    for (const auto &m : models)
    {
        print_model_iknp_report_if_p0(m, ot_perf, beaver_perf);
    }
}

struct BeaverShapeRecord
{
    std::string label;
    size_t m;
    size_t k;
    size_t n;
    size_t triples;
    size_t bytes;
};

struct BeaverOfflineStats
{
    std::vector<BeaverShapeRecord> records;
    size_t total_triples = 0;
    size_t total_bytes = 0;

    void reset()
    {
        records.clear();
        total_triples = 0;
        total_bytes = 0;
    }

    void add_matmul(const std::string &label, size_t m, size_t k, size_t n, int bit_len)
    {
        const size_t elem_bytes = (bit_len + 7) / 8;
        const size_t triples = m * k * n;
        const size_t bytes = triples * 3 * elem_bytes;
        records.push_back({label, m, k, n, triples, bytes});
        total_triples += triples;
        total_bytes += bytes;
    }

    void print_once_if_p0() const
    {
        if (Party2PC::getInstance().party_id != 0)
        {
            return;
        }
        std::cout << "[2PC Beaver Offline] shape-aware triple accounting:" << std::endl;
        for (const auto &r : records)
        {
            std::cout << "  - " << r.label << " : (M=" << r.m << ", K=" << r.k << ", N=" << r.n
                      << "), triples=" << r.triples << ", bytes=" << convert_bytes_to_string(r.bytes) << std::endl;
        }
        std::cout << "  total triples=" << total_triples
                  << ", total bytes=" << convert_bytes_to_string(total_bytes) << std::endl;
    }
};

static BeaverOfflineStats g_beaver_stats;
static size_t g_simulated_beaver_bytes = 0;

static void simulate_beaver_delta_bytes()
{
    if (g_beaver_stats.total_bytes <= g_simulated_beaver_bytes)
    {
        return;
    }
    simulate_offline_comm_bytes(g_beaver_stats.total_bytes - g_simulated_beaver_bytes);
    g_simulated_beaver_bytes = g_beaver_stats.total_bytes;
}

static size_t bytes_for_bits(int bits)
{
    return (bits + 7) / 8;
}

static void simulate_offline_comm_bytes(size_t bytes)
{
    if (bytes == 0)
    {
        return;
    }
    auto &party = Party2PC::getInstance();
    const size_t CHUNK = (size_t)1 << 20;
    std::vector<char> buf(std::min(CHUNK, bytes), 0);
    size_t rem = bytes;
    while (rem)
    {
        const size_t t = std::min(rem, buf.size());
        Tensor<char> send_t(buf.data(), {(uint32_t)t});
        Tensor<char> recv_t({(uint32_t)t});
        std::thread st([&]() { party.send_tensor_to<char>(party.next_party_id, send_t); });
        party.recv_tensor_from<char>(party.next_party_id, recv_t);
        st.join();
        rem -= t;
    }
}

// Calculate DPF key size (in bytes)
// Parameters:
//  Bin: input domain bits
//  Bout: output/bucket bits
//  groupSize: number of groups (for g array length)
// For DPF, per user's instruction we count these fields as key material:
//  block *k0, block *k1   -> each length Bin+1, each block is 16 bytes
//  GroupElement *g         -> length groupSize, element size = bytes_for_bits(Bout)
//  std::array<std::array<block,2>,2> cs -> 4 blocks

static inline size_t dpf_key_size_bytes(int Bin, int Bout, int groupSize, bool Verification = false)
{
    size_t blocks_k = Bin + 1; // k0  send to 3 parties
    size_t bytes = blocks_k * BLOCK_BYTES;
    // g array: groupSize elements of Bout bits -> count bytes
    bytes += groupSize * bytes_for_bits(Bout);
    // // r
    // bytes += groupSize * bytes_for_bits(Bin);
    if (Verification)
    {
        // Bin * g and alpha check
        bytes += 2 * groupSize * bytes_for_bits(Bin);
        // cs: 2 blocks
        bytes += 2 * BLOCK_BYTES;
    }
    return bytes;
}

// Calculate DCF key size (in bytes)
// Fields to count (per user's instruction):
//  block *k0, *k1 -> each length Bin+1
//  GroupElement *g -> length groupSize
//  GroupElement *v -> length Bin * groupSize,Vcw
//  cs: 4 blocks

static inline size_t dcf_key_size_bytes(int Bin, int Bout, int groupSize, bool Verification = false)
{
    size_t blocks_k = Bin * 2 + 1; // k0 and k1
    size_t bytes = blocks_k * BLOCK_BYTES;
    // g
    bytes += groupSize * bytes_for_bits(Bout);
    // v
    bytes += groupSize * (Bin - 1) * bytes_for_bits(Bin);
    // r
    bytes += 2 * groupSize * bytes_for_bits(Bin);
    if (Verification)
    {
        // Bin * g and alpha check
        bytes += 2 * groupSize * bytes_for_bits(Bin);
        // cs: 2 blocks
        bytes += 2 * BLOCK_BYTES;
    }
    return bytes;
}

static inline bool add_overflow_size_t(size_t a, size_t b, size_t *out)
{
    if (out == nullptr)
    {
        return true;
    }
    if (b > std::numeric_limits<size_t>::max() - a)
    {
        return true;
    }
    *out = a + b;
    return false;
}

static inline bool mul_overflow_size_t(size_t a, size_t b, size_t *out)
{
    if (out == nullptr)
    {
        return true;
    }
    if (a == 0 || b == 0)
    {
        *out = 0;
        return false;
    }
    if (a > std::numeric_limits<size_t>::max() / b)
    {
        return true;
    }
    *out = a * b;
    return false;
}

// bit_len = cmp_size,count = number of comparisons
static TwoPcDPFKeyPack offline_pointcmp(int bit_len, size_t count, bool Verification = false, size_t extra_comm = 0)
{
    omp_set_num_threads(kOfflineOmpThreads);
    auto &party = Party2PC::getInstance();
    // Use DPF key size (Bin = Bout = bit_len, groupSize = 1)
    size_t key_size = dpf_key_size_bytes(bit_len, bit_len, 1, Verification);
    size_t total_key_bytes_mul = 0;
    size_t total_key_bytes = 0;
    if (mul_overflow_size_t(count, key_size, &total_key_bytes_mul) ||
        add_overflow_size_t(total_key_bytes_mul, extra_comm, &total_key_bytes))
    {
        throw std::overflow_error("offline_pointcmp: total_key_bytes overflow");
    }

    // Generate DPF keys `count` times to simulate offline generation cost.
    // We interleave generation and communication to prevent connection timeouts
    // when count is very large (e.g. millions).
    const size_t COMM_CHUNK_SIZE = 1 << 20; // 1MB logic chunks
    // Calculate how many keys fit into roughly one CHUNK.
    // Ensure we process at least one batch of KEY_BATCH size.
    size_t keys_per_comm_chunk = 1;
    if (key_size > 0) {
        keys_per_comm_chunk = COMM_CHUNK_SIZE / key_size;
    }
    if (keys_per_comm_chunk == 0) keys_per_comm_chunk = 1;

    // Align to 256 for friendly OpenMP batching
    keys_per_comm_chunk = ((keys_per_comm_chunk + 255) / 256) * 256;
    
    // Buffer for communication simulation
    // We use total_key_bytes to size it initially if small, else clamp to CHUNK
    size_t buffer_size_alloc = std::min((size_t)COMM_CHUNK_SIZE, total_key_bytes);
    if (buffer_size_alloc == 0) buffer_size_alloc = 1024; // safety
    std::vector<char> buf(buffer_size_alloc, 0);

    size_t generated = 0;
    size_t accumulated_sim_bytes = 0;
    
    // Helper to simulate sending 'bytes' amount of data (sending twice as per original logic)
    auto simulate_send_recv = [&](size_t bytes_to_send) {
        (void)bytes_to_send;
    };

    while (generated < count)
    {
        size_t this_batch = std::min(keys_per_comm_chunk, count - generated);
        
        // 1. Compute keys (CPU simulation)
        #pragma omp parallel for
        for (long long ii = 0; ii < (long long)this_batch; ++ii)
        {
            size_t i = generated + (size_t)ii;
            GroupElement idx(0, bit_len);
            GroupElement payload(0, bit_len);
            auto keys = keyGenTwoPcDPF(bit_len, bit_len, idx, payload, GenParams(Verification));
            (void)i;
        }
        
        // 2. Communicate (Network simulation)
        // Calculate bytes for this batch
        size_t current_bytes = this_batch * key_size;
        
        // Check if we are at the end and need to cover the remainder of total_key_bytes
        // (which includes extra_comm and any alignment differences)
        if (generated + this_batch == count) {
             size_t remaining_total = total_key_bytes - accumulated_sim_bytes;
             // Use the remaining bytes if larger than current calc (handles extra_comm)
             if (remaining_total > current_bytes) {
                 current_bytes = remaining_total; 
             }
        }
        
        if (current_bytes > 0) {
            simulate_send_recv(current_bytes);
            accumulated_sim_bytes += current_bytes;
        }

        generated += this_batch;
    }
    
    // If for some reason we finished generation but didn't match total_key_bytes 
    // (e.g. if key_size calc was slightly off or extra_comm passed incorrectly),
    // handle remainder here.
    if (accumulated_sim_bytes < total_key_bytes) {
        simulate_send_recv(total_key_bytes - accumulated_sim_bytes);
    }
    // Generate one keypair to return a usable DPF key for evaluation.
    auto single_keys = keyGenTwoPcDPF(bit_len, bit_len, GroupElement(0, bit_len), GroupElement(0, bit_len), GenParams(Verification));
    // Return the first share as the representative key for eval calls.
    return std::move(single_keys.first);

    // if (Verification){
    //     DCFKeyPack kp0 = DCFKeyPack();
    //     GroupElement res(0, bit_len);
    //     GroupElement idx(0, bit_len);
    //     size_t beta = 0;
    //     size_t alaha_check = 0;
    //     // 1.Full-domain evaluation to simulate verification cost
    //     #pragma omp parallel for
    //     for (size_t i = 0; i < count; ++i) {
    //         GroupElement idx(0, bit_len);
    //         for(size_t j = 0; j < (1 << bit_len); ++j){
    //             idx = GroupElement(j, bit_len);
    //             evalDCF(party.party_id, &res, idx, kp0,Verification);
    //         }
    //         beta += res.value;
    //         alaha_check += res.value * i;
    //     }

    //     // 2. Simulate hash verification cost and Open t, s
    //     size_t hash_len = (1 << bit_len) * 2 * BLOCK_BYTES;
    //     {
    //         size_t t = hash_len + 2 * bytes_for_bits(bit_len);
    //         Tensor<char> send_t({(uint32_t)t});
    //         Tensor<char> recv_t({(uint32_t)t});
    //         std::thread st([&](){ party.send_tensor_to<char>(party.next_party_id, send_t); });
    //         party.recv_tensor_from<char>(party.next_party_id, recv_t);
    //         st.join();
    //     }

    // }
}

// bit_len = table_size
static void offline_lut(int bit_len, size_t count, bool Verification = false)
{
    omp_set_num_threads(kOfflineOmpThreads);
    auto &party = Party2PC::getInstance();
    // 1. VDPF Keys
    TwoPcDPFKeyPack kp0 = offline_pointcmp(bit_len, 3*count, Verification);
    if (not Verification)
    {
#pragma omp parallel for
        for (size_t i = 0; i < 3*count; ++i)
        {
            GroupElement res(0, bit_len);
            GroupElement idx(0, bit_len);
            for (size_t j = 0; j < (1 << bit_len); ++j)
            {
                idx = GroupElement(j, bit_len);
                evalTwoPcDPF(party.party_id, &res, idx, kp0);
            }
        }
        // free dynamically allocated buffers inside kp0
        freeTwoPcDPFKeyPack(kp0);
    }
}

static void offline_ucmp(int bit_len, size_t count, bool Verification = false, size_t extra_comm = 0)
{
    omp_set_num_threads(kOfflineOmpThreads);
    // 1. VDPF Keys
    auto &party = Party2PC::getInstance();
    // Ensure key_len is large enough to cover the range [0, 2*bit_len)
    int key_len = static_cast<int>(std::floor(std::log2(2 * bit_len))) + 1;
    
    // Generate and retrieve a usable DPF key for evaluation (instead of null pointers).
    TwoPcDPFKeyPack kp0 = offline_pointcmp(key_len, count * bit_len, Verification, extra_comm);
    // 2. eval
    #pragma omp parallel for
    for (size_t i = 0; i < 2 * bit_len; ++i)
    {
        
        GroupElement idx(0, key_len);
        for (size_t j = 0; j < count * bit_len; ++j)
        {
            idx = GroupElement(i, key_len);
            GroupElement res(0, key_len);
            // Each party performs evalVDPF to simulate eval cost
            evalTwoPcDPF(party.party_id, &res, idx, kp0);
        }
    }
    // free dynamically allocated buffers inside kp0
    freeTwoPcDPFKeyPack(kp0);
}

static void offline_drelu(int bit_len, size_t count, bool Verification = false)
{
    omp_set_num_threads(kOfflineOmpThreads);
    auto &party = Party2PC::getInstance();
    // 1. Shares of c,
    size_t share_bytes = count * bytes_for_bits(bit_len) * 2;
    offline_ucmp(bit_len, 3 * count, Verification, share_bytes); //should count * 3
    

    // 2. Reveal delta_c, delta_y
    // if (Verification){
    //     size_t a_val=0, b_val=0;
    //     SecretShareVectorMyType a(count);
    //     SecretShareVectorSmallType theta(count);
    //     // Initialize a with some shares
    //     #pragma omp parallel for
    //     for(long long ii=0; ii<(long long)count; ++ii) {
    //         size_t i = (size_t)ii;
    //         a[i].first = a_val;
    //         a[i].second = b_val;
    //     }
    //     funcWrap(a, theta, count);
    //     // if (party.party_id == 0) std::cout << "funcWrap executed successfully." << std::endl;

    //     // verify delta_c, delta_y according to protocol in the image:
    //     SecretShareTensor<size_t> delta_c_shared({(uint32_t)count});
    //     SecretShareTensor<size_t> delta_y_shared({(uint32_t)count});
    //     #pragma omp parallel for
    //     for (long long ii = 0; ii < (long long)count; ++ii) {
    //         size_t i = (size_t)ii;
    //         delta_c_shared.first.data[i] = 1 - theta[i].first;
    //         delta_c_shared.second.data[i] = 1 - theta[i].second;
    //         delta_y_shared.first.data[i] = 1 - theta[i].first;
    //         delta_y_shared.second.data[i] = 1 - theta[i].second;
    //     }
    //     __int128 total_key_bytes_128 = ( __int128)2 * ( __int128)count * ( __int128)bytes_for_bits(bit_len);
    //     if (total_key_bytes_128 > ( __int128)std::numeric_limits<size_t>::max()){
    //     } else {
    //         size_t total_key_bytes = (size_t)total_key_bytes_128;
    //         const size_t CHUNK = (size_t)1 << 20; // 1MB
    //         std::vector<char> buf(std::min(CHUNK, total_key_bytes), 0);
    //         size_t rem = total_key_bytes;
    //         while (rem) {
    //             size_t t = std::min(rem, buf.size());
    //             Tensor<char> send_t({(uint32_t)t});
    //             std::memcpy(send_t.data, buf.data(), t);
    //             Tensor<char> recv_t({(uint32_t)t});
    //             std::thread st([&](){ party.send_tensor_to<char>(party.next_party_id, send_t); });
    //             party.recv_tensor_from<char>(party.next_party_id, recv_t);
    //             st.join();
    //             rem -= t;
    //         }
    //         rem = total_key_bytes;
    //         while (rem) {
    //             size_t t = std::min(rem, buf.size());
    //             Tensor<char> send_t({(uint32_t)t});
    //             std::memcpy(send_t.data, buf.data(), t);
    //             Tensor<char> recv_t({(uint32_t)t});
    //             std::thread st([&](){ party.send_tensor_to<char>(party.next_party_id, send_t); });
    //             party.recv_tensor_from<char>(party.next_party_id, recv_t);
    //             st.join();
    //             rem -= t;
    //         }
    //     }

    // }
}

static void run_offline_softmax(int party_id,size_t exp_num, size_t max_num){
    omp_set_num_threads(kOfflineOmpThreads);
   // Softmax： uses UCMP for Max
    offline_drelu(max_last_dim_eval_bitlength(), max_num);
    // Assuming Softmax uses UCMP for e^x
    offline_drelu(11, exp_num);
    offline_lut(4, exp_num); // compute >>z
}

static void run_offline_Inv(int party_id,size_t inv_num){
    omp_set_num_threads(kOfflineOmpThreads);
// Inv: One for base LUT; One for Inv LUT
    offline_drelu(31, 2 * inv_num);
    offline_lut(2, inv_num);
    offline_lut(12, inv_num);
}

static void run_offline_MHA(int party_id,size_t exp_num, size_t inv_num, size_t max_num){
   // Softmax： uses UCMP for Max
    omp_set_num_threads(kOfflineOmpThreads);
    offline_drelu(max_last_dim_eval_bitlength(), max_num);
    // Inv: One for base LUT; One for Inv LUT
    offline_drelu(31, 2 * inv_num);
    offline_lut(2, inv_num);
    offline_lut(12, inv_num);
    // Assuming Softmax uses UCMP for e^x
    offline_drelu(11, exp_num);
    offline_lut(4, exp_num); // compute >>z

    // Shape-aware Beaver accounting for MHA core matmuls under 2PC.
    TransformerConfig config("transformer", 64);
    const size_t batch = batch_size;
    const size_t seq_len = config.seq_len;
    const size_t hidden = config.hidden_size;
    const size_t heads = config.num_attention_heads;
    const size_t head_dim = hidden / heads;
    const size_t tokens = batch * seq_len;
    const size_t bh = batch * heads;

    g_beaver_stats.add_matmul("mha.qkv_proj", tokens, hidden, 3 * hidden, 64);
    g_beaver_stats.add_matmul("mha.qk^T", bh * seq_len, head_dim, seq_len, 64);
    g_beaver_stats.add_matmul("mha.attn_v", bh * seq_len, seq_len, head_dim, 64);
    g_beaver_stats.add_matmul("mha.out_proj", tokens, hidden, hidden, 64);
    simulate_beaver_delta_bytes();

}

static void run_offline_ln(int party_id,size_t ln_num){
   // Rsqt:One for base LUT; One for rsqt LUT; One for exsqrt
    omp_set_num_threads(kOfflineOmpThreads);
    offline_drelu(31, 2 * ln_num);
    offline_lut(2, ln_num);
    offline_lut(2, ln_num);
    offline_lut(12, ln_num);

    // LN core multiplications: variance accumulation and affine scale roughly as token-wise muls.
    TransformerConfig config("transformer", 64);
    const size_t tokens = batch_size * config.seq_len;
    g_beaver_stats.add_matmul("ln.elementwise_mul_equiv", tokens, config.hidden_size, 1, 64);
    simulate_beaver_delta_bytes();
}

static void run_offline_ffn(int party_id,size_t gelu_num){
   //Gelu
    omp_set_num_threads(kOfflineOmpThreads);
    offline_drelu(15, gelu_num * 3);
    // secure_sin now consumes dealer-distributed ring shares instead of LUT/DPF keys.

    TransformerConfig config("transformer", 64);
    const size_t tokens = batch_size * config.seq_len;
    g_beaver_stats.add_matmul("ffn.up_proj", tokens, config.hidden_size, config.intermediate_size, 64);
    g_beaver_stats.add_matmul("ffn.down_proj", tokens, config.intermediate_size, config.hidden_size, 64);
    simulate_beaver_delta_bytes();
}

static void run_offline_lut_12(int party_id,size_t num){
    omp_set_num_threads(kOfflineOmpThreads);
    offline_lut(12, num);
}


static void offline_tf_test(int party_id){
    Party2PC::getInstance().load_param<uint64_t>();
    TransformerConfig tf_config("transformer", 64);
    omp_set_num_threads(kOfflineOmpThreads);
    size_t seq_len = tf_config.seq_len;
    size_t hidden = tf_config.hidden_size;
    size_t intermediate = tf_config.intermediate_size;
    size_t heads = tf_config.num_attention_heads;
    // GELU (16-bit LUT)
    // FFN structure: hidden -> intermediate -> (GELU) -> hidden
    size_t gelu_count =  seq_len * intermediate;

    // Softmax (32-bit LUT)
    // Attention scores matrix: batch * heads * seq_len * seq_len
    size_t softmax_exp_count =  heads * seq_len * seq_len;
    size_t softmax_inv_count =  heads * seq_len;

    // LayerNorm (32-bit LUT)
    // Encoder layers have 2 LNs each.
    // bert_test in llm2pc.cpp adds 1 extra SecMixedLayerNorm.
    size_t ln_count =  seq_len;

    // Softmax Max (UCMP)
    // Finding max of seq_len elements requires seq_len - 1 comparisons.
    size_t softmax_max_count =  heads * seq_len * (seq_len - 1);

    g_beaver_stats.reset();

    // for (uint32_t num = 10000; num <= 10000; num *= 10)
    // {
    //     if (party_id == 0)
    //     {
    //         std::cout << "********************** num = " << num << " **********************" << std::endl;
    //     }
    //     test_func<Party2PC>(run_offline_softmax, "Offline Softmax", loop, warmup_rounds, party_id, num, num-1);
    //     Party2PC::getInstance().sync();
    //     test_func<Party2PC>(run_offline_Inv, "Offline Inv", loop, warmup_rounds, party_id, num);
    //     Party2PC::getInstance().sync();
    //     test_func<Party2PC>(run_offline_ffn, "Offline GELU", loop, warmup_rounds, party_id, num);
    //     Party2PC::getInstance().sync();
    //     test_func<Party2PC>(run_offline_ln, "Offline Reciprocal Square Root", loop, warmup_rounds, party_id, num);
    //     Party2PC::getInstance().sync();
    //     test_func<Party2PC>(run_offline_lut_12, "Offline LUT-12", loop, warmup_rounds, party_id, num);
    //     Party2PC::getInstance().sync();
        
    // }
    test_func<Party2PC>(run_offline_MHA, "Offline MHA", loop, warmup_rounds, party_id, softmax_exp_count/2, softmax_inv_count/2, softmax_max_count/2);
    Party2PC::getInstance().sync();
    test_func<Party2PC>(run_offline_MHA, "Offline MMHA", loop, warmup_rounds, party_id, softmax_exp_count/2, softmax_inv_count/2, softmax_max_count/2/2);
    Party2PC::getInstance().sync();
    test_func<Party2PC>(run_offline_ffn, "Offline FFN", loop, warmup_rounds, party_id, gelu_count/2);
    Party2PC::getInstance().sync();
    test_func<Party2PC>(run_offline_ln, "Offline LayerNorm", loop, warmup_rounds, party_id, ln_count);
    Party2PC::getInstance().sync();
    g_beaver_stats.print_once_if_p0();
}

static void run_offline_bert(int party_id) {
    (void)party_id;
    omp_set_num_threads(kOfflineOmpThreads);
    const ModelOfflineProfile profile = build_model_offline_profile("bert-base", TransformerConfig("bert-base", 128));
    for (const auto &workload : profile.workloads)
    {
        if (workload.kind == OfflineWorkloadKind::Drelu)
        {
            offline_drelu(static_cast<int>(workload.bit_len), workload.count);
        }
        else if (workload.kind == OfflineWorkloadKind::Lut)
        {
            offline_lut(static_cast<int>(workload.bit_len), workload.count);
        }
    }
}

static void run_offline_gpt2(int party_id) {
    (void)party_id;
    omp_set_num_threads(kOfflineOmpThreads);
    const ModelOfflineProfile profile = build_model_offline_profile("gpt2", TransformerConfig("gpt2", 64));
    for (const auto &workload : profile.workloads)
    {
        if (workload.kind == OfflineWorkloadKind::Drelu)
        {
            offline_drelu(static_cast<int>(workload.bit_len), workload.count);
        }
        else if (workload.kind == OfflineWorkloadKind::Lut)
        {
            offline_lut(static_cast<int>(workload.bit_len), workload.count);
        }
    }
}

static void run_offline_llama2_16(int party_id)
{
    (void)party_id;
    omp_set_num_threads(kOfflineOmpThreads);
    const ModelOfflineProfile profile = build_model_offline_profile("llama2-7b-16tok", TransformerConfig("llama2-7b", 16));
    for (const auto &workload : profile.workloads)
    {
        if (workload.kind == OfflineWorkloadKind::Drelu)
        {
            offline_drelu(static_cast<int>(workload.bit_len), workload.count);
        }
        else if (workload.kind == OfflineWorkloadKind::Lut)
        {
            offline_lut(static_cast<int>(workload.bit_len), workload.count);
        }
    }
}

static void run_offline_llama2_128(int party_id)
{
    (void)party_id;
    omp_set_num_threads(kOfflineOmpThreads);
    const ModelOfflineProfile profile = build_model_offline_profile("llama2-7b-128tok", TransformerConfig("llama2-7b", 128));
    for (const auto &workload : profile.workloads)
    {
        if (workload.kind == OfflineWorkloadKind::Drelu)
        {
            offline_drelu(static_cast<int>(workload.bit_len), workload.count);
        }
        else if (workload.kind == OfflineWorkloadKind::Lut)
        {
            offline_lut(static_cast<int>(workload.bit_len), workload.count);
        }
    }
}

static uint64_t shaft_lut_bytes_per_party(uint32_t bit_len, uint64_t count)
{
    (void)bit_len;
    // Remove DPF/DCF key-size based accounting: model LUT offline material as dealer-distributed ring shares.
    return 0;
}

static uint64_t shaft_a2b_upper_bound_bytes_per_party(uint32_t bit_len, uint64_t count)
{
    const uint64_t effective_bit_len = static_cast<uint64_t>(effective_a2b_adder_bitlength(bit_len));
    const uint64_t boolean_triples = count * a2b_adder_and_gates(static_cast<uint32_t>(effective_bit_len));
    const uint64_t bits_per_party = boolean_triples * 3ULL;
    return (bits_per_party + 7ULL) / 8ULL;
}

static uint64_t shaft_drelu_bytes_per_party(uint32_t bit_len, uint64_t count)
{
    (void)bit_len;
    (void)count;
    // dReLU is no longer a first-class offline primitive in Shaft LLM accounting.
    // Online sign extraction is accounted via A2B/B2A, and follow-up arithmetic muls are covered by Beaver.
    return 0;
}

static uint64_t shaft_sine_mask_bytes_per_party(uint64_t count)
{
    return 3ULL * count * bytes_for_bits(static_cast<int>(RING_WORD_BITS));
}

static DealerDistributionSummary summarize_shaft_dealer_distribution(const ModelOfflineProfile &profile)
{
    DealerDistributionSummary summary;
    summary.rounds_per_party = profile.workloads.size();
    summary.total_send_actions = profile.workloads.size() * 2ULL;

    for (const auto &workload : profile.workloads)
    {
        if (workload.kind == OfflineWorkloadKind::A2b)
        {
            const uint64_t bytes = shaft_a2b_upper_bound_bytes_per_party(workload.bit_len, workload.count);
            summary.per_party_exact_bytes += bytes;
            summary.per_party_upper_bound_bytes += bytes;
            summary.total_a2b += workload.count;
            summary.total_a2b_effective_bits += workload.count * static_cast<uint64_t>(effective_a2b_adder_bitlength(workload.bit_len));
            summary.total_a2b_impl_boolean_triples += workload.count * a2b_adder_and_gates(runtime_a2b_adder_bitlength(workload.bit_len));
        }
        else if (workload.kind == OfflineWorkloadKind::Drelu)
        {
            continue;
        }
        else if (workload.kind == OfflineWorkloadKind::Lut)
        {
            summary.per_party_exact_bytes += shaft_lut_bytes_per_party(workload.bit_len, workload.count);
            summary.per_party_upper_bound_bytes += shaft_lut_bytes_per_party(workload.bit_len, workload.count);
        }
        else
        {
            const uint64_t bytes = shaft_sine_mask_bytes_per_party(workload.count);
            summary.per_party_exact_bytes += bytes;
            summary.per_party_upper_bound_bytes += bytes;
            summary.total_sine_mask += workload.count;
        }
    }

    for (const auto &call : profile.precise_a2b_calls)
    {
        summary.total_a2b_adder_invocations += call.adder_invocations;
    }

    summary.total_exact_dealer_bytes = summary.per_party_exact_bytes * 2ULL;
    summary.total_upper_bound_dealer_bytes = summary.per_party_upper_bound_bytes * 2ULL;
    return summary;
}

static BeaverDistributionSummary summarize_shaft_beaver_distribution(const ModelBeaverSummary &summary,
                                                                    const DealerBeaverPerfSample &perf)
{
    BeaverDistributionSummary beaver_summary;
    beaver_summary.total_triples = summary.total_multiplications;

    const uint64_t bytes_per_triple_per_party = 3ULL * bytes_for_bits(64);
    beaver_summary.per_party_bytes = beaver_summary.total_triples * bytes_per_triple_per_party;
    beaver_summary.total_dealer_bytes = beaver_summary.per_party_bytes * 2ULL;
    beaver_summary.rounds_per_party = beaver_summary.total_triples == 0 ? 0ULL : 1ULL;
    beaver_summary.total_send_actions = beaver_summary.rounds_per_party * 2ULL;
    beaver_summary.estimated_seconds = perf.triples_per_sec > 0.0
                                           ? static_cast<double>(beaver_summary.total_triples) / perf.triples_per_sec
                                           : 0.0;
    return beaver_summary;
}

static void offline_a2b(uint32_t bit_len, uint64_t count)
{
    // Emulate mode uses a conservative upper bound for per-party A2B payload,
    // computed with the effective comparison bitlength rather than the backend's full 64-bit ring width.
    const uint64_t bytes = shaft_a2b_upper_bound_bytes_per_party(bit_len, count);
    simulate_offline_comm_bytes(static_cast<size_t>(bytes));
}

static void offline_sine_mask(uint64_t count)
{
    // secure_sin mask material is modeled as ring-share payload from dealer.
    const uint64_t bytes = shaft_sine_mask_bytes_per_party(count);
    simulate_offline_comm_bytes(static_cast<size_t>(bytes));
}

static void run_offline_workload(int party_id, const OfflinePrimitiveWorkload &workload)
{
    (void)party_id;
    omp_set_num_threads(kOfflineOmpThreads);
    if (workload.kind == OfflineWorkloadKind::A2b)
    {
        offline_a2b(workload.bit_len, workload.count);
    }
    else if (workload.kind == OfflineWorkloadKind::Drelu)
    {
        offline_drelu(static_cast<int>(workload.bit_len), workload.count, Verification);
    }
    else if (workload.kind == OfflineWorkloadKind::Lut)
    {
        offline_lut(static_cast<int>(workload.bit_len), workload.count, Verification);
    }
    else if (workload.kind == OfflineWorkloadKind::SineMask)
    {
        offline_sine_mask(workload.count);
    }
}

template <typename P>
static void test_func_scaled_offline(const std::string &test_name,
                                     const ModelOfflineProfile &measured_profile,
                                     const ModelOfflineProfile &full_profile,
                                     int loop_count,
                                     int warmup_count,
                                     int party_id)
{
    if (P::getInstance().party_id == 0)
    {
        std::cout << "====================Test for " << test_name << " ====================" << std::endl;
    }

    if (P::getInstance().party_id == 0)
    {
        std::cout << "  runtime measurement mode=full 1-layer workload" << std::endl;
    }

    double measured_wall_total = 0.0;
    double measured_cpu_total = 0.0;

    for (const auto &workload : measured_profile.workloads)
    {
        if (P::getInstance().party_id == 0)
        {
            std::cout << "  [measure] start " << workload.label
                      << " bit_len=" << workload.bit_len
                      << " count=" << workload.count << std::endl;
        }

        for (int i = 0; i < warmup_count; ++i)
        {
            run_offline_workload(party_id, workload);
        }

        struct timespec requestStart, requestEnd;
        clock_t start = clock();
        clock_gettime(CLOCK_REALTIME, &requestStart);

        for (int i = 0; i < loop_count; ++i)
        {
            run_offline_workload(party_id, workload);
        }

        clock_t clock_end = clock();
        clock_gettime(CLOCK_REALTIME, &requestEnd);

        const double measured_wall = diff(requestStart, requestEnd) / loop_count;
        const double measured_cpu = static_cast<double>(clock_end - start) / CLOCKS_PER_SEC / loop_count;

        measured_wall_total += measured_wall;
        measured_cpu_total += measured_cpu;

        if (P::getInstance().party_id == 0)
        {
            std::cout << "  [measure] done  " << workload.label
                      << " wall=" << measured_wall << " sec" << std::endl;
        }
    }

    if (P::getInstance().party_id == 0)
    {
        const double scale = measured_profile.config.num_layers == 0
                                 ? 0.0
                                 : static_cast<double>(full_profile.config.num_layers) / measured_profile.config.num_layers;
        const DealerBeaverPerfSample beaver_perf = benchmark_third_party_dealer_generation(1 << 20);
        const DealerDistributionSummary measured_summary = summarize_shaft_dealer_distribution(measured_profile);
        const DealerDistributionSummary full_summary = summarize_shaft_dealer_distribution(full_profile);
        const BeaverDistributionSummary measured_beaver_summary = summarize_shaft_beaver_distribution(measured_profile.beaver_summary, beaver_perf);
        const BeaverDistributionSummary full_beaver_summary = summarize_shaft_beaver_distribution(full_profile.beaver_summary, beaver_perf);

        std::cout << "Time:" << std::endl;
        std::cout << "  measured wall clock total (" << measured_profile.config.num_layers << " layer): " << measured_wall_total << " sec" << std::endl;
        std::cout << "  measured CPU total (" << measured_profile.config.num_layers << " layer): " << measured_cpu_total << " sec" << std::endl;
        std::cout << "  scaled full-model wall clock (" << full_profile.config.num_layers << " layers): "
              << measured_wall_total * scale << " sec" << std::endl;
        std::cout << "  Beaver dealer benchmark: sampled " << beaver_perf.sampled_triples
                  << " triples in " << std::fixed << std::setprecision(6) << beaver_perf.seconds
                  << " s, throughput=" << std::setprecision(2) << beaver_perf.triples_per_sec << " triples/s" << std::endl;
        std::cout << "  measured Beaver dealer time (" << measured_profile.config.num_layers << " layer): "
                  << std::setprecision(3) << measured_beaver_summary.estimated_seconds << " sec (separate estimate, not included in measured wall/CPU)" << std::endl;
        std::cout << "  scaled full-model Beaver dealer time (" << full_profile.config.num_layers << " layers): "
                  << full_beaver_summary.estimated_seconds << " sec (separate estimate, not included in measured wall/CPU)" << std::endl;
        std::cout << "  measured total offline time incl. Beaver (" << measured_profile.config.num_layers << " layer): "
              << (measured_wall_total + measured_beaver_summary.estimated_seconds) << " sec" << std::endl;
        std::cout << "  scaled full-model total offline time incl. Beaver (" << full_profile.config.num_layers << " layers): "
              << (measured_wall_total * scale + full_beaver_summary.estimated_seconds) << " sec" << std::endl;

        std::cout << "Dealer Distribution (measured workload):" << std::endl;
        std::cout << "  exact per-party storage/bytes=" << convert_bytes_to_string(measured_summary.per_party_exact_bytes)
              << ", conservative upper-bound per-party bytes=" << convert_bytes_to_string(measured_summary.per_party_upper_bound_bytes)
                  << ", rounds per party=" << measured_summary.rounds_per_party
              << ", exact total dealer bytes=" << convert_bytes_to_string(measured_summary.total_exact_dealer_bytes)
              << ", conservative upper-bound total dealer bytes=" << convert_bytes_to_string(measured_summary.total_upper_bound_dealer_bytes)
                  << ", total send actions=" << measured_summary.total_send_actions << std::endl;
        std::cout << "  primitive counts: a2b=" << measured_summary.total_a2b
                  << ", sine-mask=" << measured_summary.total_sine_mask << std::endl;
          std::cout << "  A2B exact adder invocations=" << measured_summary.total_a2b_adder_invocations
              << ", paper effective bits=" << measured_summary.total_a2b_effective_bits
              << ", boolean triples(runtime adder backend)=" << measured_summary.total_a2b_impl_boolean_triples << std::endl;

        std::cout << "Dealer Distribution (scaled full model):" << std::endl;
        std::cout << "  exact per-party storage/bytes=" << convert_bytes_to_string(full_summary.per_party_exact_bytes)
              << ", conservative upper-bound per-party bytes=" << convert_bytes_to_string(full_summary.per_party_upper_bound_bytes)
                  << ", rounds per party=" << full_summary.rounds_per_party
              << ", exact total dealer bytes=" << convert_bytes_to_string(full_summary.total_exact_dealer_bytes)
              << ", conservative upper-bound total dealer bytes=" << convert_bytes_to_string(full_summary.total_upper_bound_dealer_bytes)
                  << ", total send actions=" << full_summary.total_send_actions << std::endl;
        std::cout << "  primitive counts: a2b=" << full_summary.total_a2b
                  << ", sine-mask=" << full_summary.total_sine_mask << std::endl;
          std::cout << "  A2B exact adder invocations=" << full_summary.total_a2b_adder_invocations
              << ", paper effective bits=" << full_summary.total_a2b_effective_bits
              << ", boolean triples(runtime adder backend)=" << full_summary.total_a2b_impl_boolean_triples << std::endl;

        std::cout << "Estimated Dealer Beaver Distribution (measured workload):" << std::endl;
        std::cout << "  per-party storage/bytes=" << convert_bytes_to_string(measured_beaver_summary.per_party_bytes)
                  << ", rounds per party=" << measured_beaver_summary.rounds_per_party
                  << ", total dealer bytes=" << convert_bytes_to_string(measured_beaver_summary.total_dealer_bytes)
                  << ", total send actions=" << measured_beaver_summary.total_send_actions << std::endl;
        std::cout << "  beaver triples=" << measured_beaver_summary.total_triples << std::endl;

        std::cout << "Estimated Dealer Beaver Distribution (scaled full model):" << std::endl;
        std::cout << "  per-party storage/bytes=" << convert_bytes_to_string(full_beaver_summary.per_party_bytes)
                  << ", rounds per party=" << full_beaver_summary.rounds_per_party
                  << ", total dealer bytes=" << convert_bytes_to_string(full_beaver_summary.total_dealer_bytes)
                  << ", total send actions=" << full_beaver_summary.total_send_actions << std::endl;
        std::cout << "  beaver triples=" << full_beaver_summary.total_triples << std::endl;

        std::cout << "Total Offline Dealer Distribution (measured workload, incl. Beaver):" << std::endl;
        std::cout << "  exact per-party storage/bytes=" << convert_bytes_to_string(measured_summary.per_party_exact_bytes + measured_beaver_summary.per_party_bytes)
              << ", conservative upper-bound per-party bytes=" << convert_bytes_to_string(measured_summary.per_party_upper_bound_bytes + measured_beaver_summary.per_party_bytes)
                  << ", rounds per party=" << (measured_summary.rounds_per_party + measured_beaver_summary.rounds_per_party)
              << ", exact total dealer bytes=" << convert_bytes_to_string(measured_summary.total_exact_dealer_bytes + measured_beaver_summary.total_dealer_bytes)
              << ", conservative upper-bound total dealer bytes=" << convert_bytes_to_string(measured_summary.total_upper_bound_dealer_bytes + measured_beaver_summary.total_dealer_bytes)
                  << ", total send actions=" << (measured_summary.total_send_actions + measured_beaver_summary.total_send_actions) << std::endl;

        std::cout << "Total Offline Dealer Distribution (scaled full model, incl. Beaver):" << std::endl;
        std::cout << "  exact per-party storage/bytes=" << convert_bytes_to_string(full_summary.per_party_exact_bytes + full_beaver_summary.per_party_bytes)
              << ", conservative upper-bound per-party bytes=" << convert_bytes_to_string(full_summary.per_party_upper_bound_bytes + full_beaver_summary.per_party_bytes)
                  << ", rounds per party=" << (full_summary.rounds_per_party + full_beaver_summary.rounds_per_party)
              << ", exact total dealer bytes=" << convert_bytes_to_string(full_summary.total_exact_dealer_bytes + full_beaver_summary.total_dealer_bytes)
              << ", conservative upper-bound total dealer bytes=" << convert_bytes_to_string(full_summary.total_upper_bound_dealer_bytes + full_beaver_summary.total_dealer_bytes)
                  << ", total send actions=" << (full_summary.total_send_actions + full_beaver_summary.total_send_actions) << std::endl;
    }
}

static OfflineMaterialPerfSample benchmark_shaft_dpf_keygen(uint32_t bit_len, uint64_t sampled_instances)
{
    omp_set_num_threads(kOfflineOmpThreads);
    const uint64_t sample_count = std::max<uint64_t>(sampled_instances, 1024ULL);
    const auto begin = std::chrono::steady_clock::now();

    #pragma omp parallel for
    for (long long index = 0; index < static_cast<long long>(sample_count); ++index)
    {
        GroupElement idx(0, bit_len);
        GroupElement payload(0, bit_len);
        auto keys = keyGenTwoPcDPF(static_cast<int>(bit_len), static_cast<int>(bit_len), idx, payload, GenParams(Verification));
        (void)keys;
    }

    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - begin).count();

    OfflineMaterialPerfSample sample;
    sample.sampled_instances = sample_count;
    sample.seconds = seconds;
    sample.instances_per_sec = seconds > 0.0 ? static_cast<double>(sample_count) / seconds : 0.0;
    return sample;
}

static void print_shaft_generation_benchmark_if_p0(const ModelOfflineProfile &profile)
{
    auto &party = Party2PC::getInstance();
    if (party.party_id != 0)
    {
        return;
    }

    const DealerBeaverPerfSample beaver_perf = benchmark_third_party_dealer_generation(1 << 20);
    double total_seconds = 0.0;
    uint64_t total_precise_adder_invocations = 0;
    std::map<std::string, A2BAdderPerfSample> a2b_perf_cache;

    std::cout << "====================Offline Dealer Bench for " << profile.name << " ====================" << std::endl;
    std::cout << "  mode=dealer-bench (sampled local dealer generation/distribution, no party communication)" << std::endl;
    std::cout << "  accounting keep: A2B (public boolean adder + dealer boolean Beaver triples), secure_sin mask generation, Beaver generation/distribution" << std::endl;
    std::cout << "  accounting delete: standalone dReLU keys, LUT keys, DPF/DCF key-size summaries for LLM path" << std::endl;
    std::cout << "  note: pc_msb->b2a sign lifting does not consume Beaver; if a fixed-point mask is needed, the current path uses mulConst." << std::endl;
    std::cout << "  note: Beaver estimate covers arithmetic multiplications such as mask application in select/mul, not b2a itself." << std::endl;
    std::cout << "  note: each A2B instance is modeled as a public ripple-carry adder; only AND gates consume offline boolean triples." << std::endl;
    std::cout << "  note: paper-style comparison counts A2B by effective comparison width (48/52-bit here), and the current online/offline A2B backend now executes the same requested bitwidth." << std::endl;

    if (!profile.precise_a2b_calls.empty())
    {
        std::cout << "  precise A2B call model:" << std::endl;
        for (const auto &call : profile.precise_a2b_calls)
        {
            const std::string cache_key = std::to_string(call.bit_len) + "#" + std::to_string(call.count);
            auto it = a2b_perf_cache.find(cache_key);
            if (it == a2b_perf_cache.end())
            {
                const uint64_t sampled_instances = std::min<uint64_t>(call.count == 0 ? 1024ULL : call.count, 1ULL << 18);
                it = a2b_perf_cache.emplace(cache_key, benchmark_a2b_boolean_adder_material(call.bit_len, sampled_instances)).first;
            }

            const A2BAdderPerfSample &perf = it->second;
            const uint64_t boolean_triples = call.count * a2b_adder_and_gates(perf.adder_bit_len);
            const double estimated_seconds = perf.instances_per_sec > 0.0
                                                 ? static_cast<double>(call.count) / perf.instances_per_sec
                                                 : 0.0;

            total_seconds += estimated_seconds;
            total_precise_adder_invocations += call.adder_invocations;

            std::cout << "    " << call.label
                      << " kind=" << offline_workload_kind_name(call.kind)
                      << " eval_bit_len=" << call.bit_len
                      << " adder_bit_len=" << perf.adder_bit_len
                      << " shape=" << call.shape
                      << " instances=" << call.count
                      << " boolean_triples=" << boolean_triples
                      << " exact_adder_invocations=" << call.adder_invocations
                      << " sample=" << perf.sampled_instances
                      << " sample_boolean_triples=" << perf.sampled_boolean_triples
                      << " sample_time=" << std::fixed << std::setprecision(6) << perf.seconds << " sec"
                      << " throughput=" << std::setprecision(2) << perf.instances_per_sec << " adder-a2b/s"
                      << " triple_throughput=" << std::setprecision(2) << perf.boolean_triples_per_sec << " triples/s"
                      << " estimate=" << std::setprecision(3) << estimated_seconds << " sec"
                      << " (dealer boolean-triple generation/distribution)" << std::endl;
        }
        std::cout << "  exact total A2B adder invocations=" << total_precise_adder_invocations << std::endl;
    }

    for (const auto &workload : profile.workloads)
    {
        uint64_t total_instances = 0;
        if (workload.kind == OfflineWorkloadKind::A2b)
        {
            total_instances = workload.count * static_cast<uint64_t>(effective_a2b_adder_bitlength(workload.bit_len));
        }
        else if (workload.kind == OfflineWorkloadKind::Drelu)
        {
            total_instances = 3ULL * workload.count * workload.bit_len;
        }
        else if (workload.kind == OfflineWorkloadKind::Lut)
        {
            total_instances = 3ULL * workload.count;
        }

        if (is_legacy_offline_workload_kind(workload.kind))
        {
            std::cout << "  " << workload.label
                      << " kind=" << offline_workload_kind_name(workload.kind)
                      << " skipped (legacy offline accounting removed from Shaft LLM path)" << std::endl;
            continue;
        }

        if (workload.kind == OfflineWorkloadKind::SineMask)
        {
            const uint64_t share_words = 3ULL * workload.count;
            const uint64_t sampled_instances = std::min<uint64_t>(workload.count == 0 ? 1024ULL : workload.count, 1ULL << 20);
            const OfflineMaterialPerfSample perf = benchmark_sine_mask_generation(sampled_instances);
            const double estimated_seconds = perf.instances_per_sec > 0.0
                                                 ? static_cast<double>(workload.count) / perf.instances_per_sec
                                                 : 0.0;
            total_seconds += estimated_seconds;
            std::cout << "  " << workload.label
                      << " kind=" << offline_workload_kind_name(workload.kind)
                      << " share_words=" << share_words
                      << " sample=" << perf.sampled_instances
                      << " sample_time=" << std::fixed << std::setprecision(6) << perf.seconds << " sec"
                      << " throughput=" << std::setprecision(2) << perf.instances_per_sec << " masks/s"
                      << " estimate=" << std::setprecision(3) << estimated_seconds << " sec"
                      << " (dealer ring-share generation)" << std::endl;
            continue;
        }

        if (workload.kind == OfflineWorkloadKind::A2b)
        {
            const uint64_t boolean_triples = workload.count * a2b_adder_and_gates(effective_a2b_adder_bitlength(workload.bit_len));
            std::cout << "  " << workload.label
                      << " kind=" << offline_workload_kind_name(workload.kind)
                      << " instances=" << workload.count
                      << " boolean_triples=" << boolean_triples
                      << " exact_adder_invocations=" << workload.adder_invocations
                      << " estimate=see precise A2B call model above" << std::endl;
        }
        else
        {
            std::cout << "  " << workload.label
                      << " kind=" << offline_workload_kind_name(workload.kind)
                      << " instances=" << total_instances
                      << " estimate=n/a (DPF/DCF key statistics removed; dealer-distributed material only)" << std::endl;
        }
    }

    const BeaverDistributionSummary beaver_summary = summarize_shaft_beaver_distribution(profile.beaver_summary, beaver_perf);
    total_seconds += beaver_summary.estimated_seconds;

    std::cout << "  beaver triples=" << beaver_summary.total_triples
              << " sample=" << beaver_perf.sampled_triples
              << " sample_time=" << std::fixed << std::setprecision(6) << beaver_perf.seconds << " sec"
              << " throughput=" << std::setprecision(2) << beaver_perf.triples_per_sec << " triples/s"
              << " estimate=" << std::setprecision(3) << beaver_summary.estimated_seconds << " sec"
              << " (multi-threaded generation, OpenMP threads=" << kOfflineOmpThreads << ")" << std::endl;
    std::cout << "  total local dealer-bench estimate (including Beaver generation)=" << std::setprecision(3)
              << total_seconds << " sec" << std::endl;
}

static void print_shaft_analysis_report_if_p0(const ModelOfflineProfile &profile)
{
    auto &party = Party2PC::getInstance();
    if (party.party_id != 0)
    {
        return;
    }

    const DealerDistributionSummary summary = summarize_shaft_dealer_distribution(profile);
    const DealerBeaverPerfSample beaver_perf = benchmark_third_party_dealer_generation(1 << 20);
    const BeaverDistributionSummary beaver_summary = summarize_shaft_beaver_distribution(profile.beaver_summary, beaver_perf);

    std::cout << "====================Offline Analysis for " << profile.name << " ====================" << std::endl;
    std::cout << "  mode=analysis (formula only, no party communication)" << std::endl;
    std::cout << "  layers=" << profile.config.num_layers
              << ", seq=" << profile.config.seq_len
              << ", hidden=" << profile.config.hidden_size
              << ", intermediate=" << profile.config.intermediate_size
              << ", heads=" << profile.config.num_attention_heads << std::endl;
    std::cout << "  primitive counts: a2b=" << summary.total_a2b
              << ", sine-mask=" << summary.total_sine_mask << std::endl;
    std::cout << "  A2B exact adder invocations=" << summary.total_a2b_adder_invocations
              << ", paper effective bits=" << summary.total_a2b_effective_bits
              << ", boolean triples(runtime adder backend)=" << summary.total_a2b_impl_boolean_triples << std::endl;
    std::cout << "Dealer Distribution:" << std::endl;
    std::cout << "  exact per-party storage/bytes=" << convert_bytes_to_string(summary.per_party_exact_bytes)
              << ", conservative upper-bound per-party bytes=" << convert_bytes_to_string(summary.per_party_upper_bound_bytes)
              << ", rounds per party=" << summary.rounds_per_party
              << ", exact total dealer bytes=" << convert_bytes_to_string(summary.total_exact_dealer_bytes)
              << ", conservative upper-bound total dealer bytes=" << convert_bytes_to_string(summary.total_upper_bound_dealer_bytes)
              << ", total send actions=" << summary.total_send_actions << std::endl;
    std::cout << "Estimated Dealer Beaver Distribution:" << std::endl;
    std::cout << "  per-party storage/bytes=" << convert_bytes_to_string(beaver_summary.per_party_bytes)
              << ", rounds per party=" << beaver_summary.rounds_per_party
              << ", total dealer bytes=" << convert_bytes_to_string(beaver_summary.total_dealer_bytes)
              << ", total send actions=" << beaver_summary.total_send_actions << std::endl;
    std::cout << "  beaver triples=" << beaver_summary.total_triples
              << ", dealer generation estimate=" << std::fixed << std::setprecision(3)
              << beaver_summary.estimated_seconds << " sec (separate estimate, not included in measured wall/CPU)" << std::endl;
    std::cout << "Total Offline Dealer Distribution (including Beaver distribution):" << std::endl;
    std::cout << "  exact per-party storage/bytes=" << convert_bytes_to_string(summary.per_party_exact_bytes + beaver_summary.per_party_bytes)
              << ", conservative upper-bound per-party bytes=" << convert_bytes_to_string(summary.per_party_upper_bound_bytes + beaver_summary.per_party_bytes)
              << ", rounds per party=" << (summary.rounds_per_party + beaver_summary.rounds_per_party)
              << ", exact total dealer bytes=" << convert_bytes_to_string(summary.total_exact_dealer_bytes + beaver_summary.total_dealer_bytes)
              << ", conservative upper-bound total dealer bytes=" << convert_bytes_to_string(summary.total_upper_bound_dealer_bytes + beaver_summary.total_dealer_bytes)
              << ", total send actions=" << (summary.total_send_actions + beaver_summary.total_send_actions) << std::endl;
}

static std::string selected_offline_model()
{
    const char *selected_model = std::getenv("SHAFT_OFFLINE_LLM_MODEL");
    return selected_model == nullptr ? std::string("all") : std::string(selected_model);
}

static bool should_run_offline_model(const std::string &model_name);

static OfflineExecutionMode selected_offline_mode()
{
    const char *selected_mode = std::getenv("SHAFT_OFFLINE_MODE");
    if (selected_mode == nullptr)
    {
        return OfflineExecutionMode::Local;
    }

    const std::string mode(selected_mode);
    if (mode == "analysis")
    {
        return OfflineExecutionMode::Analysis;
    }
    if (mode == "dealer-bench")
    {
        return OfflineExecutionMode::DealerBench;
    }
    if (mode == "emulate")
    {
        return OfflineExecutionMode::Emulate;
    }
    return OfflineExecutionMode::Local;
}

static bool offline_mode_uses_network(OfflineExecutionMode mode)
{
    return mode == OfflineExecutionMode::Emulate;
}

static void run_shaft_offline_case_if_selected(const std::string &selector_a,
                                               const std::string &selector_b,
                                               const std::string &test_name,
                                               const ModelOfflineProfile &measured_profile,
                                               const ModelOfflineProfile &full_profile,
                                               OfflineExecutionMode mode,
                                               int party_id)
{
    if (!should_run_offline_model(selector_a) && !should_run_offline_model(selector_b))
    {
        return;
    }

    if (mode == OfflineExecutionMode::Analysis || mode == OfflineExecutionMode::Local)
    {
        print_shaft_analysis_report_if_p0(full_profile);
        if (mode == OfflineExecutionMode::Analysis)
        {
            return;
        }
    }

    if (mode == OfflineExecutionMode::DealerBench || mode == OfflineExecutionMode::Local)
    {
        (void)test_name;
        (void)measured_profile;
        (void)party_id;
        print_shaft_generation_benchmark_if_p0(full_profile);
        return;
    }

    test_func_scaled_offline<Party2PC>(test_name, measured_profile, full_profile, loop, warmup_rounds, party_id);
    if (offline_mode_uses_network(mode))
    {
        Party2PC::getInstance().sync();
    }
}

static bool should_run_offline_model(const std::string &model_name)
{
    const std::string selected = selected_offline_model();
    return selected == "all" || selected == model_name;
}

static bool run_full_offline_workloads_enabled()
{
    const char *value = std::getenv("SHAFT_OFFLINE_LLM_RUN_FULL");
    if (value == nullptr)
    {
        return false;
    }
    const std::string v(value);
    return v == "1" || v == "true" || v == "TRUE" || v == "on" || v == "ON";
}

void OfflineLLM2pcTest::run(int party_id)
{
    const OfflineExecutionMode mode = selected_offline_mode();
    if (offline_mode_uses_network(mode))
    {
        Test2PC::run(party_id);
        return;
    }

    if (party_id != 0 && party_id != 1)
    {
        std::cerr << "Invalid party id for 2PC" << std::endl;
        return;
    }

    Party2PC &party = Party2PC::getInstance();
    party.party_id = party_id;
    party.next_party_id = (party_id == 0) ? 1 : 0;

    if (party_id == 0)
    {
        std::cout << "P0 [local offline mode]" << std::endl;
    }
    else
    {
        std::cout << "P1 [local offline mode]" << std::endl;
    }

    main_test(party_id);

    party.party_id = -1;
    party.next_party_id = -1;
}

void OfflineLLM2pcTest::main_test(int party_id)
{
    fss_init();
    omp_set_num_threads(kOfflineOmpThreads);
    g_beaver_stats.reset();
    g_simulated_beaver_bytes = 0;
    const OfflineExecutionMode mode = selected_offline_mode();

    const TransformerConfig bert_full("bert-base", resolved_offline_seq_len(128));
    const TransformerConfig gpt2_full("gpt2", resolved_offline_seq_len(64));
    const TransformerConfig llama16_full("llama2-7b", resolved_offline_seq_len(16));
    const TransformerConfig llama128_full("llama2-7b", resolved_offline_seq_len(128));

    const ModelOfflineProfile bert_measured = build_model_offline_profile("bert-base", make_measured_offline_config(bert_full));
    const ModelOfflineProfile bert_profile = build_model_offline_profile("bert-base", bert_full);
    const ModelOfflineProfile gpt2_measured = build_model_offline_profile("gpt2", make_measured_offline_config(gpt2_full));
    const ModelOfflineProfile gpt2_profile = build_model_offline_profile("gpt2", gpt2_full);
    const ModelOfflineProfile llama16_measured = build_model_offline_profile("llama2-7b-16tok", make_measured_offline_config(llama16_full));
    const ModelOfflineProfile llama16_profile = build_model_offline_profile("llama2-7b-16tok", llama16_full);
    const ModelOfflineProfile llama128_measured = build_model_offline_profile("llama2-7b-128tok", make_measured_offline_config(llama128_full));
    const ModelOfflineProfile llama128_profile = build_model_offline_profile("llama2-7b-128tok", llama128_full);

    if (Party2PC::getInstance().party_id == 0)
    {
        std::cout << "[offline_llm2pc] mode=";
        if (mode == OfflineExecutionMode::Analysis)
        {
            std::cout << "analysis";
        }
        else if (mode == OfflineExecutionMode::DealerBench)
        {
            std::cout << "dealer-bench";
        }
        else if (mode == OfflineExecutionMode::Emulate)
        {
            std::cout << "emulate";
        }
        else
        {
            std::cout << "local";
        }
        std::cout << std::endl;
    }

    run_shaft_offline_case_if_selected("bert", "bert-base", "Offline BERT (1-layer measured)", bert_measured, bert_profile, mode, party_id);
    run_shaft_offline_case_if_selected("gpt2", "gpt2", "Offline GPT2 (1-layer measured)", gpt2_measured, gpt2_profile, mode, party_id);
    run_shaft_offline_case_if_selected("llama2-16", "llama2-7b-16tok", "Offline Llama2-7B 16tok (1-layer measured)", llama16_measured, llama16_profile, mode, party_id);
    run_shaft_offline_case_if_selected("llama2-128", "llama2-7b-128tok", "Offline Llama2-7B 128tok (1-layer measured)", llama128_measured, llama128_profile, mode, party_id);
}
