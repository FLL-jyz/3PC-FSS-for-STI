#include "party2pc.h"
#include "twopc_protocols.h"
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
#include <random>
#include <omp.h>


static uint32_t batch_size = 1;
static bool print_res = false;
static int loop = 1;
static int warmup_rounds = 0;
static constexpr size_t BLOCK_BYTES = 16;
static bool Verification = false;
static constexpr int OFFLINE_NUM_THREADS = 8;

static constexpr size_t IKNP_SECURITY_PARAM = 128;
static constexpr size_t IKNP_BLOCK_BYTES = 16;
static constexpr size_t IKNP_BOOLEAN_COT_PER_TRIPLE = 2;
static constexpr size_t RING_WORD_BITS = 64;
static constexpr size_t OFFLINE_KEYGEN_BATCH_PER_THREAD = 32;

static void simulate_offline_comm_bytes(size_t bytes);

static inline void configure_offline_parallelism()
{
    omp_set_dynamic(0);
    omp_set_num_threads(OFFLINE_NUM_THREADS);
}

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
    Drelu,
    Lut,
};

struct OfflinePrimitiveWorkload
{
    OfflineWorkloadKind kind;
    std::string label;
    uint32_t bit_len;
    uint64_t count;
};

struct ModelOfflineProfile
{
    std::string name;
    TransformerConfig config;
    std::vector<OfflinePrimitiveWorkload> workloads;
    ModelBeaverSummary beaver_summary;
};

struct DealerDistributionSummary
{
    uint64_t per_party_bytes = 0;
    uint64_t total_dealer_bytes = 0;
    uint64_t rounds_per_party = 0;
    uint64_t total_send_actions = 0;
    uint64_t total_drelu = 0;
    uint64_t total_lut = 0;
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

struct IKNPPerfSample
{
    size_t sampled_ots = 0;
    double seconds = 0.0;
    double ot_per_sec = 0.0;
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

struct OfflineCommPerfSample
{
    uint64_t bytes = 0;
    double seconds = 0.0;
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
    workloads.push_back({kind, label, bit_len, count});
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

    const uint64_t activation_count = layers * tokens * intermediate;
    const uint64_t softmax_sites = layers * batch * heads * seq_len * seq_len;
    const uint64_t softmax_inv_count = layers * batch * heads * seq_len;
    const uint64_t softmax_max_count = layers * batch * heads * seq_len * (seq_len - 1);
    const uint64_t norm_vector_count = (2 * layers + 1) * tokens;

    ModelOfflineProfile profile{model_name, config, {}, build_model_summary(model_name, config)};

    if (is_llama)
    {
        append_workload(profile.workloads, OfflineWorkloadKind::Drelu, "silu.select_relu", 62, activation_count);
        append_workload(profile.workloads, OfflineWorkloadKind::Drelu, "silu.select_relu_truncated", 47, activation_count);
        append_workload(profile.workloads, OfflineWorkloadKind::Drelu, "silu.select_abs", 47, activation_count);
        append_workload(profile.workloads, OfflineWorkloadKind::Lut, "silu.lut", lut_index_bitlength(4ULL * (1ULL << SWIGLU_TABLE_PRECISION)), activation_count);
    }
    else
    {
        append_workload(profile.workloads, OfflineWorkloadKind::Drelu, "gelu.select_relu", 62, activation_count);
        append_workload(profile.workloads, OfflineWorkloadKind::Drelu, "gelu.select_relu_truncated", 47, activation_count);
        append_workload(profile.workloads, OfflineWorkloadKind::Drelu, "gelu.select_abs", 47, activation_count);
        append_workload(profile.workloads, OfflineWorkloadKind::Lut, "gelu.lut", lut_index_bitlength(4ULL * (1ULL << GELU_TABLE_PRECISION)), activation_count);
    }

    append_workload(profile.workloads, OfflineWorkloadKind::Drelu, "softmax.max_last_dim.select", max_last_dim_eval_bitlength(), softmax_max_count);
    append_workload(profile.workloads, OfflineWorkloadKind::Drelu, "softmax.neg_exp.clip_nonnegative", 52, softmax_sites);
    append_workload(profile.workloads, OfflineWorkloadKind::Lut, "softmax.neg_exp.exp_hi", 8, softmax_sites);
    append_workload(profile.workloads, OfflineWorkloadKind::Lut, "softmax.neg_exp.exp_lo", 8, softmax_sites);
    append_workload(profile.workloads, OfflineWorkloadKind::Lut, "softmax.inv_chopped", inv_chopped_lut_bitlength(4096), softmax_inv_count);

    append_workload(profile.workloads, OfflineWorkloadKind::Drelu, "norm.rsqrt.getk", 62, norm_vector_count * rsqrt_getk_compare_fanout());
    append_workload(profile.workloads, OfflineWorkloadKind::Lut, "norm.rsqrt_shift", lut_index_bitlength(sizeof(uint64_t) * 8 + 1), norm_vector_count);
    append_workload(profile.workloads, OfflineWorkloadKind::Lut, "norm.rsqrt", 13, norm_vector_count);

    return profile;
}

static TransformerConfig make_measured_offline_config(const TransformerConfig &config)
{
    TransformerConfig measured = config;
    measured.num_layers = std::min<uint32_t>(config.num_layers, kOfflineMeasuredLayers);
    return measured;
}

static IKNPPerfSample benchmark_iknp_extension(size_t sampled_ots)
{
    configure_offline_parallelism();
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

    #pragma omp parallel for schedule(static)
    for (size_t row = 0; row < k; ++row)
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

    #pragma omp parallel for schedule(static)
    for (size_t col = 0; col < n; ++col)
    {
        uint64_t *qt_col = &qt[col * words_per_col];
        for (size_t row = 0; row < k; ++row)
        {
            const size_t row_base = row * words_per_row;
            const uint64_t word = q[row_base + (col >> 6)];
            const uint64_t bit = (word >> (col & 63)) & 1ULL;
            if (bit)
            {
                qt_col[row >> 6] |= (1ULL << (row & 63));
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
    configure_offline_parallelism();
    const size_t n = std::max<size_t>(sampled_triples, 1024);
    IKNPExtensionFactory iknp;
    iknp.extend_ots(n * IKNP_BOOLEAN_COT_PER_TRIPLE);

    uint64_t checksum = 0;
    const auto begin = std::chrono::steady_clock::now();

    #pragma omp parallel reduction(+:checksum)
    {
        std::mt19937_64 rng(0xBEEF1234ULL + static_cast<uint64_t>(omp_get_thread_num()));
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i)
        {
            const BoolBeaverShares tri = generate_boolean_beaver_triple_iknp(rng);
            checksum += tri.c1 ^ tri.c2 ^ static_cast<uint64_t>(i & 1ULL);
        }
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

static DealerBeaverPerfSample benchmark_third_party_dealer_generation(size_t sampled_triples)
{
    configure_offline_parallelism();
    const size_t n = std::max<size_t>(sampled_triples, 1024);
    uint64_t checksum = 0;

    const auto begin = std::chrono::steady_clock::now();
    #pragma omp parallel reduction(^:checksum)
    {
        std::mt19937_64 rng(0x12345678ULL + static_cast<uint64_t>(omp_get_thread_num()));
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i)
        {
            // Trusted dealer directly samples a,b in ring and computes c=a*b.
            const uint64_t a = rng();
            const uint64_t b = rng();
            const uint64_t c = a * b;
            checksum ^= c ^ static_cast<uint64_t>(i);
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
              << " s, throughput=" << std::setprecision(2) << perf.triples_per_sec << " triples/s" << std::endl;
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

    uint64_t total_drelu = 0;
    uint64_t total_lut = 0;
    std::cout << "[Offline Protocol Report] Model=" << profile.name
              << " (layers=" << profile.config.num_layers
              << ", seq=" << profile.config.seq_len
              << ", hidden=" << profile.config.hidden_size
              << ", intermediate=" << profile.config.intermediate_size
              << ", heads=" << profile.config.num_attention_heads << ")" << std::endl;
    for (const auto &workload : profile.workloads)
    {
        std::cout << "  " << (workload.kind == OfflineWorkloadKind::Drelu ? "offline_drelu" : "offline_lut")
                  << "(" << workload.bit_len << ")"
                  << "  label=" << workload.label
                  << ", count=" << workload.count << std::endl;
        if (workload.kind == OfflineWorkloadKind::Drelu)
        {
            total_drelu += workload.count;
        }
        else
        {
            total_lut += workload.count;
        }
    }
    std::cout << "  total offline_drelu count=" << total_drelu << std::endl;
    std::cout << "  total offline_lut count=" << total_lut << std::endl;
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

// Calculate 2PC DPF key size (in bytes).
// Parameters:
//  Bin: input domain bits
//  Bout: output/bucket bits
//  groupSize: number of groups (for g array length)
// TwoPcDPFKeyPack contains:
//  block *k               -> length Bin+1, each block is 16 bytes
//  GroupElement *g        -> length groupSize, element size = bytes_for_bits(Bout)
//  GroupElement *v        -> length Bin * groupSize, element size = bytes_for_bits(Bout)
//  cs tags                -> generated only for verification mode

static inline size_t dpf_key_size_bytes(int Bin, int Bout, int groupSize, bool Verification = false)
{
    size_t blocks_k = Bin + 1;
    size_t bytes = blocks_k * BLOCK_BYTES;
    bytes += groupSize * bytes_for_bits(Bout);
    bytes += groupSize * Bin * bytes_for_bits(Bout);
    if (Verification)
    {
        bytes += 4 * BLOCK_BYTES;
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
    configure_offline_parallelism();
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

    const size_t max_parallel_batch = static_cast<size_t>(OFFLINE_NUM_THREADS) * OFFLINE_KEYGEN_BATCH_PER_THREAD;
    keys_per_comm_chunk = std::min(keys_per_comm_chunk, max_parallel_batch);

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
        #pragma omp parallel for schedule(static)
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
    configure_offline_parallelism();
    auto &party = Party2PC::getInstance();
    // One LUT instance consumes one offline DPF key in the local estimator.
    TwoPcDPFKeyPack kp0 = offline_pointcmp(bit_len, count, Verification);
    if (not Verification)
    {
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < count; ++i)
        {
            GroupElement res(0, bit_len);
            GroupElement idx(0, bit_len);
            for (size_t j = 0; j < (1 << bit_len); ++j)
            {
                idx = GroupElement(j, bit_len);
                groto_dpf::evalGrotoDPF(party.party_id, &res, idx, kp0, false);
            }
        }
        // free dynamically allocated buffers inside kp0
        freeTwoPcDPFKeyPack(kp0);
    }
}

static void offline_ucmp(int bit_len, size_t count, bool Verification = false, size_t extra_comm = 0)
{
    configure_offline_parallelism();
    auto &party = Party2PC::getInstance();
    // Match the online path in params.h/twopc_protocols.h: pc_cmp keys are generated
    // directly with eval_bitlength instead of a compressed log2(2n)+1 domain.
    const int key_len = bit_len;

    // Match the accounting path: one compare instance contributes one offline key.
    TwoPcDPFKeyPack kp0 = offline_pointcmp(key_len, count, Verification, extra_comm);
    #pragma omp parallel for schedule(static)
    for (long long ii = 0; ii < static_cast<long long>(count); ++ii)
    {
        const auto eval_point = static_cast<uint64_t>(ii) % static_cast<uint64_t>(2 * bit_len);
        GroupElement idx(eval_point, key_len);
        GroupElement res(0, 1);
        groto_dpf::evalGrotoDPF(party.party_id, &res, idx, kp0, false);
    }
    freeTwoPcDPFKeyPack(kp0);
}

static void offline_drelu(int bit_len, size_t count, bool Verification = false)
{
    size_t share_bytes = count * bytes_for_bits(bit_len);
    offline_ucmp(bit_len, count, Verification, share_bytes);
    

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
    configure_offline_parallelism();
   // Softmax： uses UCMP for Max
    offline_drelu(max_last_dim_eval_bitlength(), max_num);
    offline_drelu(52, exp_num);
    offline_lut(8, exp_num);
    offline_lut(8, exp_num);
}

static void run_offline_Inv(int party_id,size_t inv_num){
    configure_offline_parallelism();
    (void)party_id;
    offline_lut(static_cast<int>(inv_chopped_lut_bitlength(4096)), inv_num);
}

static void run_offline_MHA(int party_id,size_t exp_num, size_t inv_num, size_t max_num){
   // Softmax： uses UCMP for Max
    configure_offline_parallelism();
    offline_drelu(max_last_dim_eval_bitlength(), max_num);
    offline_lut(static_cast<int>(inv_chopped_lut_bitlength(4096)), inv_num);
    offline_drelu(52, exp_num);
    offline_lut(8, exp_num);
    offline_lut(8, exp_num);

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
    // rsqrt: getk compare chain + shift LUT + rsqrt LUT
    configure_offline_parallelism();
     offline_drelu(62, ln_num * rsqrt_getk_compare_fanout());
     offline_lut(static_cast<int>(lut_index_bitlength(sizeof(uint64_t) * 8 + 1)), ln_num);
     offline_lut(13, ln_num);

    // LN core multiplications: variance accumulation and affine scale roughly as token-wise muls.
    TransformerConfig config("transformer", 64);
    const size_t tokens = batch_size * config.seq_len;
    g_beaver_stats.add_matmul("ln.elementwise_mul_equiv", tokens, config.hidden_size, 1, 64);
    simulate_beaver_delta_bytes();
}

static void run_offline_ffn(int party_id,size_t gelu_num){
   //Gelu
    configure_offline_parallelism();
    offline_drelu(62, gelu_num);
    offline_drelu(47, gelu_num * 2);
    offline_lut(static_cast<int>(lut_index_bitlength(4ULL * (1ULL << GELU_TABLE_PRECISION))), gelu_num);

    TransformerConfig config("transformer", 64);
    const size_t tokens = batch_size * config.seq_len;
    g_beaver_stats.add_matmul("ffn.up_proj", tokens, config.hidden_size, config.intermediate_size, 64);
    g_beaver_stats.add_matmul("ffn.down_proj", tokens, config.intermediate_size, config.hidden_size, 64);
    simulate_beaver_delta_bytes();
}

static void run_offline_lut_12(int party_id,size_t num){
    configure_offline_parallelism();
    offline_lut(12, num);
}


static void offline_tf_test(int party_id){
    Party2PC::getInstance().load_param<uint64_t>();
    TransformerConfig tf_config("transformer", 64);
    configure_offline_parallelism();
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
    configure_offline_parallelism();
    const ModelOfflineProfile profile = build_model_offline_profile("bert-base", TransformerConfig("bert-base", 128));
    for (const auto &workload : profile.workloads)
    {
        if (workload.kind == OfflineWorkloadKind::Drelu)
        {
            offline_drelu(static_cast<int>(workload.bit_len), workload.count);
        }
        else
        {
            offline_lut(static_cast<int>(workload.bit_len), workload.count);
        }
    }
}

static void run_offline_gpt2(int party_id) {
    (void)party_id;
    configure_offline_parallelism();
    const ModelOfflineProfile profile = build_model_offline_profile("gpt2", TransformerConfig("gpt2", 64));
    for (const auto &workload : profile.workloads)
    {
        if (workload.kind == OfflineWorkloadKind::Drelu)
        {
            offline_drelu(static_cast<int>(workload.bit_len), workload.count);
        }
        else
        {
            offline_lut(static_cast<int>(workload.bit_len), workload.count);
        }
    }
}

static void run_offline_llama2_16(int party_id)
{
    (void)party_id;
    configure_offline_parallelism();
    const ModelOfflineProfile profile = build_model_offline_profile("llama2-7b-16tok", TransformerConfig("llama2-7b", 16));
    for (const auto &workload : profile.workloads)
    {
        if (workload.kind == OfflineWorkloadKind::Drelu)
        {
            offline_drelu(static_cast<int>(workload.bit_len), workload.count);
        }
        else
        {
            offline_lut(static_cast<int>(workload.bit_len), workload.count);
        }
    }
}

static void run_offline_llama2_128(int party_id)
{
    (void)party_id;
    configure_offline_parallelism();
    const ModelOfflineProfile profile = build_model_offline_profile("llama2-7b-128tok", TransformerConfig("llama2-7b", 128));
    for (const auto &workload : profile.workloads)
    {
        if (workload.kind == OfflineWorkloadKind::Drelu)
        {
            offline_drelu(static_cast<int>(workload.bit_len), workload.count);
        }
        else
        {
            offline_lut(static_cast<int>(workload.bit_len), workload.count);
        }
    }
}

static uint32_t sigma_compare_key_bitlength(uint32_t bit_len)
{
    return bit_len;
}

static uint64_t sigma_lut_bytes_per_party(uint32_t bit_len, uint64_t count)
{
    return count * dpf_key_size_bytes(static_cast<int>(bit_len), static_cast<int>(bit_len), 1, Verification);
}

static uint64_t sigma_drelu_bytes_per_party(uint32_t bit_len, uint64_t count)
{
    const uint64_t share_bytes = count * bytes_for_bits(static_cast<int>(bit_len));
    const int key_len = static_cast<int>(sigma_compare_key_bitlength(bit_len));
    return count * dpf_key_size_bytes(key_len, key_len, 1, Verification) + share_bytes;
}

static DealerDistributionSummary summarize_sigma_dealer_distribution(const ModelOfflineProfile &profile)
{
    DealerDistributionSummary summary;
    summary.rounds_per_party = profile.workloads.size();
    summary.total_send_actions = profile.workloads.size() * 2ULL;

    for (const auto &workload : profile.workloads)
    {
        if (workload.kind == OfflineWorkloadKind::Drelu)
        {
            summary.per_party_bytes += sigma_drelu_bytes_per_party(workload.bit_len, workload.count);
            summary.total_drelu += workload.count;
        }
        else
        {
            summary.per_party_bytes += sigma_lut_bytes_per_party(workload.bit_len, workload.count);
            summary.total_lut += workload.count;
        }
    }

    summary.total_dealer_bytes = summary.per_party_bytes * 2ULL;
    return summary;
}

static BeaverDistributionSummary summarize_sigma_beaver_distribution(const ModelBeaverSummary &summary,
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

static OfflineCommPerfSample benchmark_offline_distribution_bytes(uint64_t bytes,
                                                                 int loop_count,
                                                                 int warmup_count)
{
    OfflineCommPerfSample sample;
    sample.bytes = bytes;
    if (bytes == 0 || loop_count <= 0)
    {
        return sample;
    }

    configure_offline_parallelism();
    auto &party = Party2PC::getInstance();
    for (int i = 0; i < warmup_count; ++i)
    {
        simulate_offline_comm_bytes(static_cast<size_t>(bytes));
        party.sync();
    }

    struct timespec requestStart, requestEnd;
    clock_gettime(CLOCK_REALTIME, &requestStart);
    for (int i = 0; i < loop_count; ++i)
    {
        simulate_offline_comm_bytes(static_cast<size_t>(bytes));
        party.sync();
    }
    clock_gettime(CLOCK_REALTIME, &requestEnd);
    sample.seconds = diff(requestStart, requestEnd) / loop_count;
    return sample;
}

static void run_offline_workload(int party_id, const OfflinePrimitiveWorkload &workload)
{
    (void)party_id;
    configure_offline_parallelism();
    if (workload.kind == OfflineWorkloadKind::Drelu)
    {
        offline_drelu(static_cast<int>(workload.bit_len), workload.count, Verification);
    }
    else
    {
        offline_lut(static_cast<int>(workload.bit_len), workload.count, Verification);
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

    const double scale = measured_profile.config.num_layers == 0
                             ? 0.0
                             : static_cast<double>(full_profile.config.num_layers) / measured_profile.config.num_layers;
    const DealerBeaverPerfSample beaver_perf = benchmark_third_party_dealer_generation(1 << 20);
    const DealerDistributionSummary measured_summary = summarize_sigma_dealer_distribution(measured_profile);
    const DealerDistributionSummary full_summary = summarize_sigma_dealer_distribution(full_profile);
    const BeaverDistributionSummary measured_beaver_summary = summarize_sigma_beaver_distribution(measured_profile.beaver_summary, beaver_perf);
    const BeaverDistributionSummary full_beaver_summary = summarize_sigma_beaver_distribution(full_profile.beaver_summary, beaver_perf);
    const OfflineCommPerfSample measured_beaver_distribution_perf = benchmark_offline_distribution_bytes(
        measured_beaver_summary.per_party_bytes, loop_count, warmup_count);

    if (P::getInstance().party_id == 0)
    {
        const double scaled_beaver_distribution_seconds = measured_beaver_distribution_perf.seconds * scale;
        const double measured_total_offline_seconds = measured_wall_total + measured_beaver_distribution_perf.seconds + measured_beaver_summary.estimated_seconds;
        const double scaled_total_offline_seconds = measured_wall_total * scale + scaled_beaver_distribution_seconds + full_beaver_summary.estimated_seconds;

        std::cout << "Time:" << std::endl;
        std::cout << "  measured wall clock total (" << measured_profile.config.num_layers << " layer): " << measured_wall_total << " sec" << std::endl;
        std::cout << "  measured CPU total (" << measured_profile.config.num_layers << " layer): " << measured_cpu_total << " sec" << std::endl;
        std::cout << "  scaled full-model wall clock (" << full_profile.config.num_layers << " layers): "
              << measured_wall_total * scale << " sec" << std::endl;
        std::cout << "  Beaver dealer benchmark: sampled " << beaver_perf.sampled_triples
                  << " triples in " << std::fixed << std::setprecision(6) << beaver_perf.seconds
                  << " s, throughput=" << std::setprecision(2) << beaver_perf.triples_per_sec << " triples/s" << std::endl;
        std::cout << "  measured Beaver dealer time (" << measured_profile.config.num_layers << " layer): "
                  << std::setprecision(3) << measured_beaver_summary.estimated_seconds << " sec" << std::endl;
        std::cout << "  scaled full-model Beaver dealer time (" << full_profile.config.num_layers << " layers): "
                  << full_beaver_summary.estimated_seconds << " sec" << std::endl;
        std::cout << "  measured Beaver distribution time (" << measured_profile.config.num_layers << " layer): "
              << measured_beaver_distribution_perf.seconds << " sec" << std::endl;
        std::cout << "  scaled full-model Beaver distribution time (" << full_profile.config.num_layers << " layers): "
              << scaled_beaver_distribution_seconds << " sec" << std::endl;
        std::cout << "  measured total offline time incl. Beaver gen+dist (" << measured_profile.config.num_layers << " layer): "
              << measured_total_offline_seconds << " sec" << std::endl;
        std::cout << "  scaled total offline time incl. Beaver gen+dist (" << full_profile.config.num_layers << " layers): "
              << scaled_total_offline_seconds << " sec" << std::endl;

        std::cout << "Dealer Distribution (measured workload):" << std::endl;
        std::cout << "  per-party storage/bytes=" << convert_bytes_to_string(measured_summary.per_party_bytes)
                  << ", rounds per party=" << measured_summary.rounds_per_party
                  << ", total dealer bytes=" << convert_bytes_to_string(measured_summary.total_dealer_bytes)
                  << ", total send actions=" << measured_summary.total_send_actions << std::endl;
        std::cout << "  primitive counts: drelu=" << measured_summary.total_drelu
                  << ", lut=" << measured_summary.total_lut << std::endl;

        std::cout << "Dealer Distribution (scaled full model):" << std::endl;
        std::cout << "  per-party storage/bytes=" << convert_bytes_to_string(full_summary.per_party_bytes)
                  << ", rounds per party=" << full_summary.rounds_per_party
                  << ", total dealer bytes=" << convert_bytes_to_string(full_summary.total_dealer_bytes)
                  << ", total send actions=" << full_summary.total_send_actions << std::endl;
        std::cout << "  primitive counts: drelu=" << full_summary.total_drelu
                  << ", lut=" << full_summary.total_lut << std::endl;

        std::cout << "Dealer Beaver Distribution (measured workload):" << std::endl;
        std::cout << "  per-party storage/bytes=" << convert_bytes_to_string(measured_beaver_summary.per_party_bytes)
                  << ", rounds per party=" << measured_beaver_summary.rounds_per_party
                  << ", total dealer bytes=" << convert_bytes_to_string(measured_beaver_summary.total_dealer_bytes)
                  << ", total send actions=" << measured_beaver_summary.total_send_actions << std::endl;
        std::cout << "  beaver triples=" << measured_beaver_summary.total_triples << std::endl;

        std::cout << "Dealer Beaver Distribution (scaled full model):" << std::endl;
        std::cout << "  per-party storage/bytes=" << convert_bytes_to_string(full_beaver_summary.per_party_bytes)
                  << ", rounds per party=" << full_beaver_summary.rounds_per_party
                  << ", total dealer bytes=" << convert_bytes_to_string(full_beaver_summary.total_dealer_bytes)
                  << ", total send actions=" << full_beaver_summary.total_send_actions << std::endl;
        std::cout << "  beaver triples=" << full_beaver_summary.total_triples << std::endl;

        std::cout << "Combined Dealer Distribution (measured workload):" << std::endl;
        std::cout << "  per-party storage/bytes=" << convert_bytes_to_string(measured_summary.per_party_bytes + measured_beaver_summary.per_party_bytes)
                  << ", rounds per party=" << (measured_summary.rounds_per_party + measured_beaver_summary.rounds_per_party)
                  << ", total dealer bytes=" << convert_bytes_to_string(measured_summary.total_dealer_bytes + measured_beaver_summary.total_dealer_bytes)
                  << ", total send actions=" << (measured_summary.total_send_actions + measured_beaver_summary.total_send_actions) << std::endl;

        std::cout << "Combined Dealer Distribution (scaled full model):" << std::endl;
        std::cout << "  per-party storage/bytes=" << convert_bytes_to_string(full_summary.per_party_bytes + full_beaver_summary.per_party_bytes)
                  << ", rounds per party=" << (full_summary.rounds_per_party + full_beaver_summary.rounds_per_party)
                  << ", total dealer bytes=" << convert_bytes_to_string(full_summary.total_dealer_bytes + full_beaver_summary.total_dealer_bytes)
                  << ", total send actions=" << (full_summary.total_send_actions + full_beaver_summary.total_send_actions) << std::endl;
    }
}

static OfflineMaterialPerfSample benchmark_sigma_dpf_keygen(uint32_t bit_len, uint64_t sampled_instances)
{
    configure_offline_parallelism();
    const uint64_t sample_count = std::max<uint64_t>(sampled_instances, 1024ULL);
    const auto begin = std::chrono::steady_clock::now();

    #pragma omp parallel for schedule(static)
    for (uint64_t index = 0; index < sample_count; ++index)
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

static void print_sigma_generation_benchmark_if_p0(const ModelOfflineProfile &profile)
{
    auto &party = Party2PC::getInstance();
    if (party.party_id != 0)
    {
        return;
    }

    const DealerBeaverPerfSample beaver_perf = benchmark_third_party_dealer_generation(1 << 20);
    double total_seconds = 0.0;

    std::cout << "====================Offline Dealer Bench for " << profile.name << " ====================" << std::endl;
    std::cout << "  mode=dealer-bench (sampled local key generation, no party communication)" << std::endl;

    for (const auto &workload : profile.workloads)
    {
        uint64_t total_instances = 0;
        uint32_t key_bit_len = workload.bit_len;

        if (workload.kind == OfflineWorkloadKind::Drelu)
        {
            key_bit_len = sigma_compare_key_bitlength(workload.bit_len);
            total_instances = workload.count;
        }
        else
        {
            total_instances = workload.count;
        }

        const uint64_t sampled_instances = std::min<uint64_t>(total_instances == 0 ? 1024ULL : total_instances, 8192ULL);
        const OfflineMaterialPerfSample perf = benchmark_sigma_dpf_keygen(key_bit_len, sampled_instances);
        const double estimated_seconds = perf.instances_per_sec > 0.0
                                             ? static_cast<double>(total_instances) / perf.instances_per_sec
                                             : 0.0;
        total_seconds += estimated_seconds;

        std::cout << "  " << workload.label
                  << " kind=" << (workload.kind == OfflineWorkloadKind::Drelu ? "drelu" : "lut")
                  << " key_bit_len=" << key_bit_len
                  << " instances=" << total_instances
                  << " sample=" << perf.sampled_instances
                  << " sample_time=" << std::fixed << std::setprecision(6) << perf.seconds << " sec"
                  << " throughput=" << std::setprecision(2) << perf.instances_per_sec << " keys/s"
                  << " estimate=" << std::setprecision(3) << estimated_seconds << " sec" << std::endl;
    }

    const BeaverDistributionSummary beaver_summary = summarize_sigma_beaver_distribution(profile.beaver_summary, beaver_perf);
    total_seconds += beaver_summary.estimated_seconds;

    std::cout << "  beaver triples=" << beaver_summary.total_triples
              << " sample=" << beaver_perf.sampled_triples
              << " sample_time=" << std::fixed << std::setprecision(6) << beaver_perf.seconds << " sec"
              << " throughput=" << std::setprecision(2) << beaver_perf.triples_per_sec << " triples/s"
              << " estimate=" << std::setprecision(3) << beaver_summary.estimated_seconds << " sec" << std::endl;
    std::cout << "  total local dealer-bench estimate=" << std::setprecision(3) << total_seconds << " sec" << std::endl;
}

static void print_sigma_analysis_report_if_p0(const ModelOfflineProfile &profile)
{
    auto &party = Party2PC::getInstance();
    if (party.party_id != 0)
    {
        return;
    }

    const DealerDistributionSummary summary = summarize_sigma_dealer_distribution(profile);
    const DealerBeaverPerfSample beaver_perf = benchmark_third_party_dealer_generation(1 << 20);
    const BeaverDistributionSummary beaver_summary = summarize_sigma_beaver_distribution(profile.beaver_summary, beaver_perf);

    std::cout << "====================Offline Analysis for " << profile.name << " ====================" << std::endl;
    std::cout << "  mode=analysis (formula only, no party communication)" << std::endl;
    std::cout << "  layers=" << profile.config.num_layers
              << ", seq=" << profile.config.seq_len
              << ", hidden=" << profile.config.hidden_size
              << ", intermediate=" << profile.config.intermediate_size
              << ", heads=" << profile.config.num_attention_heads << std::endl;
    std::cout << "  primitive counts: drelu=" << summary.total_drelu
              << ", lut=" << summary.total_lut << std::endl;
    std::cout << "Dealer Distribution:" << std::endl;
    std::cout << "  per-party storage/bytes=" << convert_bytes_to_string(summary.per_party_bytes)
              << ", rounds per party=" << summary.rounds_per_party
              << ", total dealer bytes=" << convert_bytes_to_string(summary.total_dealer_bytes)
              << ", total send actions=" << summary.total_send_actions << std::endl;
    std::cout << "Dealer Beaver Distribution:" << std::endl;
    std::cout << "  per-party storage/bytes=" << convert_bytes_to_string(beaver_summary.per_party_bytes)
              << ", rounds per party=" << beaver_summary.rounds_per_party
              << ", total dealer bytes=" << convert_bytes_to_string(beaver_summary.total_dealer_bytes)
              << ", total send actions=" << beaver_summary.total_send_actions << std::endl;
    std::cout << "  beaver triples=" << beaver_summary.total_triples
              << ", dealer generation estimate=" << std::fixed << std::setprecision(3)
              << beaver_summary.estimated_seconds << " sec" << std::endl;
    std::cout << "Combined Dealer Distribution:" << std::endl;
    std::cout << "  per-party storage/bytes=" << convert_bytes_to_string(summary.per_party_bytes + beaver_summary.per_party_bytes)
              << ", rounds per party=" << (summary.rounds_per_party + beaver_summary.rounds_per_party)
              << ", total dealer bytes=" << convert_bytes_to_string(summary.total_dealer_bytes + beaver_summary.total_dealer_bytes)
              << ", total send actions=" << (summary.total_send_actions + beaver_summary.total_send_actions) << std::endl;
}

static std::string selected_offline_model()
{
    const char *selected_model = std::getenv("SIGMA_OFFLINE_LLM_MODEL");
    return selected_model == nullptr ? std::string("all") : std::string(selected_model);
}

static bool should_run_offline_model(const std::string &model_name);

static OfflineExecutionMode selected_offline_mode()
{
    const char *selected_mode = std::getenv("SIGMA_OFFLINE_MODE");
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

static void run_sigma_offline_case_if_selected(const std::string &selector_a,
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
        print_sigma_analysis_report_if_p0(full_profile);
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
        print_sigma_generation_benchmark_if_p0(full_profile);
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
    const char *value = std::getenv("SIGMA_OFFLINE_LLM_RUN_FULL");
    if (value == nullptr)
    {
        return false;
    }
    const std::string v(value);
    return v == "1" || v == "true" || v == "TRUE" || v == "on" || v == "ON";
}

void OfflineLLM2pcTest::run(int party_id)
{
    configure_offline_parallelism();
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
    g_beaver_stats.reset();
    g_simulated_beaver_bytes = 0;
    const OfflineExecutionMode mode = selected_offline_mode();

    const TransformerConfig bert_full("bert-base", 128);
    const TransformerConfig gpt2_full("gpt2", 64);
    const TransformerConfig llama16_full("llama2-7b", 16);
    const TransformerConfig llama128_full("llama2-7b", 128);

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

    run_sigma_offline_case_if_selected("bert", "bert-base", "Offline BERT (1-layer measured)", bert_measured, bert_profile, mode, party_id);
    run_sigma_offline_case_if_selected("gpt2", "gpt2", "Offline GPT2 (1-layer measured)", gpt2_measured, gpt2_profile, mode, party_id);
    run_sigma_offline_case_if_selected("llama2-16", "llama2-7b-16tok", "Offline Llama2-7B 16tok (1-layer measured)", llama16_measured, llama16_profile, mode, party_id);
    run_sigma_offline_case_if_selected("llama2-128", "llama2-7b-128tok", "Offline Llama2-7B 128tok (1-layer measured)", llama128_measured, llama128_profile, mode, party_id);
}
