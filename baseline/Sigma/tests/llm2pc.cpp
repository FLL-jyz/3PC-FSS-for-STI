#include "party2pc.h"
#include "twopc_protocols.h"
#include "layers.h"
#include "llmConfig.h"
#include "params.h"
#include "perf_stats.h"
#include "test_base.h"
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <omp.h>
#include <string>

static uint32_t batch_size = 1;
static bool print_res = false;
static int loop = 3;
static int warmup_rounds = 2;
static int llm_loop = 3;
static int llm_warmup_rounds = 2;

static void print_llama_profile_breakdown(double wall_per_layer)
{
    const auto llama_stats = llama_profile::snapshot();
    const auto softmax_stats = twopc_protocols::profile::snapshot();

    const auto ns_to_ms = [](uint64_t ns) -> double {
        return static_cast<double>(ns) / 1e6;
    };
    const auto ns_to_s = [](uint64_t ns) -> double {
        return static_cast<double>(ns) / 1e9;
    };
    const auto avg_ms = [&](uint64_t ns, uint64_t calls) -> double {
        return calls == 0 ? 0.0 : ns_to_ms(ns) / static_cast<double>(calls);
    };
    const auto ratio = [&](uint64_t ns) -> double {
        if (wall_per_layer <= 0.0)
        {
            return 0.0;
        }
        return ns_to_s(ns) / wall_per_layer * 100.0;
    };

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Llama single-layer profile (party0):" << std::endl;
    std::cout << "  attention_total: " << ns_to_ms(llama_stats.attention_ns) << " ms"
              << " (calls=" << llama_stats.attention_calls
              << ", avg=" << avg_ms(llama_stats.attention_ns, llama_stats.attention_calls) << " ms, "
              << ratio(llama_stats.attention_ns) << "% of wall)" << std::endl;
    std::cout << "  softmax_total:   " << ns_to_ms(softmax_stats.softmax_ns) << " ms"
              << " (calls=" << softmax_stats.softmax_calls
              << ", avg=" << avg_ms(softmax_stats.softmax_ns, softmax_stats.softmax_calls) << " ms, "
              << ratio(softmax_stats.softmax_ns) << "% of wall)" << std::endl;
    std::cout << "  swiglu_total:    " << ns_to_ms(llama_stats.swiglu_ns) << " ms"
              << " (calls=" << llama_stats.swiglu_calls
              << ", avg=" << avg_ms(llama_stats.swiglu_ns, llama_stats.swiglu_calls) << " ms, "
              << ratio(llama_stats.swiglu_ns) << "% of wall)" << std::endl;
    std::cout << "  rmsnorm_total:   " << ns_to_ms(llama_stats.rmsnorm_ns) << " ms"
              << " (calls=" << llama_stats.rmsnorm_calls
              << ", avg=" << avg_ms(llama_stats.rmsnorm_ns, llama_stats.rmsnorm_calls) << " ms, "
              << ratio(llama_stats.rmsnorm_ns) << "% of wall)" << std::endl;
}

static bool quick_mode_enabled()
{
    const char *v = std::getenv("SIGMA_LLM2PC_QUICK");
    if (v == nullptr)
    {
        return false;
    }

    std::string value(v);
    return value == "1" || value == "true" || value == "TRUE" || value == "on" || value == "ON" || value == "yes" || value == "YES";
}

static bool quick_verify = quick_mode_enabled();

static std::string selected_model()
{
    const char *value = std::getenv("SIGMA_LLM2PC_MODEL");
    return value == nullptr ? std::string() : std::string(value);
}

static bool should_run_model(const char *model_name)
{
    const std::string selected = selected_model();
    return selected.empty() || selected == model_name;
}

static void tf_test(int party_id);

static void tf_test_64(int party_id);


static void bert_test(int party_id);

static void gpt2_test(int party_id);

static void gpt2_xl_test(int party_id);

static void silu_test(int party_id);

static void rmsnorm_test(int party_id);

static void llama2_test(int party_id);

static TransformerConfig make_single_layer_llama2_config(int seq_len)
{
    TransformerConfig config("llama2-7b", seq_len);
    config.num_layers = 1;
    return config;
}

void LLM2pcTest::main_test(int party_id)
{
    perf_stats::set_max_depth(1);
    perf_stats::enable(true);
    // llama2_test(party_id);
    // Party2PC::getInstance().sync();
    // tf_test_64(party_id);
    // Party2PC::getInstance().sync();
    // if (party_id == 0)
    // {
    //     std::cout << "[LLM2PC] Start silu" << std::endl;
    // }
    // silu_test(party_id);
    // Party2PC::getInstance().sync();

    // if (party_id == 0)
    // {
    //     std::cout << "[LLM2PC] Start rmsnorm" << std::endl;
    // }
    // rmsnorm_test(party_id);
    // Party2PC::getInstance().sync();

    if (party_id == 0)
    {
        std::cout << "[LLM2PC] Start llama2-7b" << std::endl;
    }
    if (should_run_model("llama2") || should_run_model("llama2-7b"))
    {
        llama2_test(party_id);
        Party2PC::getInstance().sync();
    }

    if (party_id == 0 && (should_run_model("bert") || should_run_model("bert-base") || selected_model().empty()))
    {
        std::cout << "[LLM2PC] Start bert-base" << std::endl;
    }
    if (should_run_model("bert") || should_run_model("bert-base"))
    {
        bert_test(party_id);
        Party2PC::getInstance().sync();
    }

    if (party_id == 0 && (should_run_model("gpt2") || selected_model().empty()))
    {
        std::cout << "[LLM2PC] Start gpt2" << std::endl;
    }
    if (should_run_model("gpt2"))
    {
        gpt2_test(party_id);
        Party2PC::getInstance().sync();
    }

    if (party_id == 0)
    {
        std::cout << "[LLM2PC] All done" << std::endl;
    }
}

static void share_data(twopc_protocols::ShareTensor<uint64_t> &x_share)
{
    x_share.share.rand(-5, 5);

    Tensor<uint64_t>::mul(x_share.share, x_share.share, 1 << FLOAT_PRECISION_64);
}

static void recv_share(twopc_protocols::ShareTensor<uint64_t> &x_share)
{
    x_share.zeros();
}

static void mixed_ffn_forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters, Layers<uint64_t> &layer)
{
    layer(x, parameters);
}

static void mixed_layer_forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters, Layers<uint64_t> &layer)
{
    layer(x, parameters);
}

static void mixed_llm_forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters, Layers<uint64_t> &layer)
{
    layer(x, parameters);
    twopc_protocols::macCheckSimulate<uint64_t>(MAC_SIZE);
}

static void mixed_llm_forward_ln(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters,
                                 Layers<uint64_t> &layer, Layers<uint64_t> &layer_norm)
{
    layer_norm(x, parameters);
    layer(x, parameters);
    twopc_protocols::macCheckSimulate<uint64_t>(MAC_SIZE);
}

static void mixed_gpt2_forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters, Layers<uint64_t> &layer)
{
    layer(x, parameters);
    twopc_protocols::macCheckSimulate<uint64_t>(MAC_SIZE);
}

static void layer_forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters, Layers<uint64_t> &layer)
{
    layer(x, parameters);
}

static void llm_forward(twopc_protocols::ShareTensor<uint64_t> &x, Parameters<uint64_t> &parameters, Layers<uint64_t> &layer)
{
    layer(x, parameters);
    twopc_protocols::macCheckSimulate<uint64_t>(MAC_SIZE);
}

static void benchmark_single_layer_and_estimate_full(int full_num_layers,
                                                     int benchmark_loop,
                                                     int warmup_loop,
                                                     twopc_protocols::ShareTensor<uint64_t> &x,
                                                     Parameters<uint64_t> &parameters,
                                                     Layers<uint64_t> &layer)
{
    for (int i = 0; i < warmup_loop; ++i)
    {
        mixed_llm_forward(x, parameters, layer);
    }

    llama_profile::reset();
    twopc_protocols::profile::reset();
    if (perf_stats::is_enabled())
    {
        perf_stats::reset();
    }

    Party2PC &party = Party2PC::getInstance();
    const uint64_t rounds_sent_before = party.rounds_sent();
    const uint64_t rounds_recv_before = party.rounds_received();
    const uint64_t bytes_sent_before = party.bytes_sent();
    const uint64_t bytes_recv_before = party.bytes_received();

    const auto wall_begin = std::chrono::steady_clock::now();
    const clock_t cpu_begin = clock();

    for (int i = 0; i < benchmark_loop; ++i)
    {
        mixed_llm_forward(x, parameters, layer);
    }

    const clock_t cpu_end = clock();
    const auto wall_end = std::chrono::steady_clock::now();

    if (party.party_id == 0)
    {
        const double wall_per_layer = std::chrono::duration<double>(wall_end - wall_begin).count() / benchmark_loop;
        const double cpu_per_layer = static_cast<double>(cpu_end - cpu_begin) / CLOCKS_PER_SEC / benchmark_loop;

        const uint64_t sent_rounds_per_layer = (party.rounds_sent() - rounds_sent_before) / benchmark_loop;
        const uint64_t recv_rounds_per_layer = (party.rounds_received() - rounds_recv_before) / benchmark_loop;
        const uint64_t sent_bytes_per_layer = (party.bytes_sent() - bytes_sent_before) / benchmark_loop;
        const uint64_t recv_bytes_per_layer = (party.bytes_received() - bytes_recv_before) / benchmark_loop;

        std::cout << "====================Test for llama2-7b single-layer====================" << std::endl;
        std::cout << "Time:" << std::endl;
        std::cout << "Wall Clock time: " << wall_per_layer << " sec\n";
        std::cout << "CPU time: " << cpu_per_layer << std::endl;
        std::cout << "Scaled by layers (x" << full_num_layers << "):" << std::endl;
        std::cout << "Wall Clock time: " << wall_per_layer * full_num_layers << " sec\n";
        std::cout << "CPU time: " << cpu_per_layer * full_num_layers << std::endl;

        std::cout << "Comm.: " << std::endl;
        std::cout << "\t Sent rounds:" << sent_rounds_per_layer << "\t Sent Bytes:"
                  << convert_bytes_to_string(sent_bytes_per_layer) << std::endl;
        std::cout << "\t Recv rounds:" << recv_rounds_per_layer << "\t Recv Bytes:"
                  << convert_bytes_to_string(recv_bytes_per_layer) << std::endl;

        std::cout << "Scaled by layers (x" << full_num_layers << "):" << std::endl;
        std::cout << "\t Sent rounds:" << sent_rounds_per_layer * full_num_layers
                  << "\t Sent Bytes:" << convert_bytes_to_string(sent_bytes_per_layer * full_num_layers) << std::endl;
        std::cout << "\t Recv rounds:" << recv_rounds_per_layer * full_num_layers
                  << "\t Recv Bytes:" << convert_bytes_to_string(recv_bytes_per_layer * full_num_layers) << std::endl;

        print_llama_profile_breakdown(wall_per_layer);
        if (perf_stats::is_enabled())
        {
            perf_stats::print_report(std::cout, static_cast<uint64_t>(benchmark_loop));
        }
    }
}

static void tf_test(int party_id)
{
    Party2PC::getInstance().load_param<uint64_t>();
    TransformerConfig tf_config("transformer", 64);
    Layers<uint64_t> *attention = new SecMixedAttention(tf_config);
    Layers<uint64_t> *masked_attention = new SecMixedAttention(tf_config, true);
    Layers<uint64_t> *ffn = new SecMixedFFN(tf_config);
    Layers<uint64_t> *layer_norm = new SecMixedLayerNorm(tf_config.hidden_size);
    Layers<uint64_t> *encoder = new SecMixedEncoder(tf_config);
    Layers<uint64_t> *decoder = new SecMixedDecoder(tf_config);

    std::vector<uint32_t> x_shape = {batch_size, tf_config.seq_len, tf_config.hidden_size};

    omp_set_num_threads(16);
    twopc_protocols::ShareTensor<uint64_t> x_share(x_shape);
    if (party_id == 0)
    {
        share_data(x_share);
    }
    else
    {
        recv_share(x_share);
    }

    Parameters<uint64_t> parameters;
    parameters.init_all();

    test_func<Party2PC>(mixed_layer_forward, "attention", loop, warmup_rounds, x_share, parameters, (*attention));
    test_func<Party2PC>(mixed_layer_forward, "masked attention", loop, warmup_rounds, x_share, parameters, (*masked_attention));
    test_func<Party2PC>(mixed_ffn_forward, "ffn", loop, warmup_rounds, x_share, parameters, (*ffn));
    test_func<Party2PC>(mixed_layer_forward, "layernorm", loop, warmup_rounds, x_share, parameters, (*layer_norm));
    twopc_protocols::macCheckSimulate<uint64_t>(MAC_SIZE);
    test_func<Party2PC>(mixed_llm_forward, "encoder", loop, warmup_rounds, x_share, parameters, (*encoder));
    test_func<Party2PC>(mixed_llm_forward, "decoder", loop, warmup_rounds, x_share, parameters, (*decoder));
}

static void tf_test_64(int party_id)
{
    Party2PC::getInstance().load_param<uint64_t>();
    TransformerConfig tf_config("transformer", 64);
    Layers<uint64_t> *attention = new SecAttention<uint64_t>(tf_config);
    Layers<uint64_t> *masked_attention = new SecAttention<uint64_t>(tf_config, true);
    Layers<uint64_t> *ffn = new SecFFN<uint64_t>(tf_config);
    Layers<uint64_t> *layer_norm = new SecLayerNorm<uint64_t>(tf_config.hidden_size);
    Layers<uint64_t> *encoder = new SecEncoder<uint64_t>(tf_config);
    Layers<uint64_t> *decoder = new SecDecoder<uint64_t>(tf_config);

    std::vector<uint32_t> x_shape = {batch_size, tf_config.seq_len, tf_config.hidden_size};

    omp_set_num_threads(16);
    twopc_protocols::ShareTensor<uint64_t> x_share(x_shape);
    if (party_id == 0)
    {
        share_data(x_share);
    }
    else
    {
        recv_share(x_share);
    }

    Parameters<uint64_t> parameters;
    parameters.init_all();

    test_func<Party2PC>(layer_forward, "attention 64", loop, warmup_rounds, x_share, parameters, (*attention));
    test_func<Party2PC>(layer_forward, "masked attention 64", loop, warmup_rounds, x_share, parameters, (*masked_attention));
    test_func<Party2PC>(layer_forward, "ffn 64", loop, warmup_rounds, x_share, parameters, (*ffn));
    test_func<Party2PC>(layer_forward, "layernorm 64", loop, warmup_rounds, x_share, parameters, (*layer_norm));
    twopc_protocols::macCheckSimulate<uint64_t>(MAC_SIZE);
}

static void bert_test(int party_id)
{
    Party2PC::getInstance().load_param<uint64_t>();
    TransformerConfig bert_config("bert-base", 128);
    Layers<uint64_t> *bert = new SecMixedEncoder(bert_config);

    std::vector<uint32_t> x_shape = {batch_size, bert_config.seq_len, bert_config.hidden_size};

    omp_set_num_threads(8);
    twopc_protocols::ShareTensor<uint64_t> x_share(x_shape);
    if (party_id == 0)
    {   
        share_data(x_share);
    }
    else
    {   
        recv_share(x_share);
    }

    Parameters<uint64_t> parameters;
    parameters.init_all();
    Layers<uint64_t> *layer_norm = new SecMixedLayerNorm(bert_config.hidden_size);

    if (quick_verify)
    {
        if (party_id == 0)
        {
            std::cout << "[LLM2PC][QUICK] bert-base config/load check passed" << std::endl;
        }
        Party2PC::getInstance().sync();
        return;
    }

    test_func<Party2PC>(mixed_llm_forward_ln, "bert-base", llm_loop, llm_warmup_rounds, x_share, parameters, (*bert), (*layer_norm));
}

static void gpt2_test(int party_id)
{
    Party2PC::getInstance().load_param<uint64_t>();
    TransformerConfig gpt2_config("gpt2", 64);
    Layers<uint64_t> *gpt2 = new SecMixedGPT2(gpt2_config);

    std::vector<uint32_t> x_shape = {batch_size, gpt2_config.seq_len, gpt2_config.hidden_size};

    omp_set_num_threads(8);
    twopc_protocols::ShareTensor<uint64_t> x_share(x_shape);
    if (party_id == 0)
    {
        share_data(x_share);
    }
    else
    {
        recv_share(x_share);
    }

    Parameters<uint64_t> parameters;
    parameters.init_all();

    if (quick_verify)
    {
        if (party_id == 0)
        {
            std::cout << "[LLM2PC][QUICK] gpt2 config/load check passed" << std::endl;
        }
        Party2PC::getInstance().sync();
        return;
    }

    test_func<Party2PC>(mixed_gpt2_forward, "gpt2", llm_loop, llm_warmup_rounds, x_share, parameters, (*gpt2));
}

static void gpt2_xl_test(int party_id)
{
    Party2PC::getInstance().load_param<uint64_t>();
    TransformerConfig gpt2xl_config("gpt2-XL", 128);
    Layers<uint64_t> *gpt2 = new SecMixedGPT2(gpt2xl_config);

    std::vector<uint32_t> x_shape = {batch_size, gpt2xl_config.seq_len, gpt2xl_config.hidden_size};

    omp_set_num_threads(1);
    twopc_protocols::ShareTensor<uint64_t> x_share(x_shape);
    if (party_id == 0)
    {   
        share_data(x_share);
    }
    else
    {   
        recv_share(x_share);
    }

    Parameters<uint64_t> parameters;
    parameters.init_all();

    test_func<Party2PC>(mixed_gpt2_forward, "gpt2-xl", loop, warmup_rounds, x_share, parameters, (*gpt2));
}

static void silu_test(int party_id)
{
    Party2PC::getInstance().load_param<uint64_t>();
    std::vector<uint32_t> x_shape = {batch_size, 8, 4096};
    Layers<uint64_t> *silu = new SecSiLU<uint64_t>();

    omp_set_num_threads(8);
    twopc_protocols::ShareTensor<uint64_t> x_share(x_shape);
    if (party_id == 0)
    {
        share_data(x_share);
    }
    else
    {
        recv_share(x_share);
    }

    Parameters<uint64_t> parameters;
    parameters.init_all();

    if (quick_verify)
    {
        if (party_id == 0)
        {
            std::cout << "[LLM2PC][QUICK] silu config/load check passed" << std::endl;
        }
        Party2PC::getInstance().sync();
        return;
    }

    test_func<Party2PC>(mixed_layer_forward, "silu", llm_loop, llm_warmup_rounds, x_share, parameters, (*silu));
}

static void rmsnorm_test(int party_id)
{
    Party2PC::getInstance().load_param<uint64_t>();
    std::vector<uint32_t> x_shape = {batch_size, 8, 4096};
    Layers<uint64_t> *rmsnorm = new SecMixedRMSNorm(4096);

    omp_set_num_threads(8);
    twopc_protocols::ShareTensor<uint64_t> x_share(x_shape);
    if (party_id == 0)
    {
        share_data(x_share);
    }
    else
    {
        recv_share(x_share);
    }

    Parameters<uint64_t> parameters;
    parameters.init_all();

    if (quick_verify)
    {
        if (party_id == 0)
        {
            std::cout << "[LLM2PC][QUICK] rmsnorm config/load check passed" << std::endl;
        }
        Party2PC::getInstance().sync();
        return;
    }

    test_func<Party2PC>(mixed_layer_forward, "rmsnorm", llm_loop, llm_warmup_rounds, x_share, parameters, (*rmsnorm));
}

static void llama2_test(int party_id)
{
    Party2PC::getInstance().load_param<uint64_t>();
    TransformerConfig full_llama2_config("llama2-7b", 128);
    TransformerConfig llama2_config = make_single_layer_llama2_config(128);
    Layers<uint64_t> *llama2 = new SecMixedLlama2(llama2_config);

    std::vector<uint32_t> x_shape = {batch_size, llama2_config.seq_len, llama2_config.hidden_size};
    omp_set_num_threads(8);
    twopc_protocols::ShareTensor<uint64_t> x_share(x_shape);
    if (party_id == 0)
    {
        share_data(x_share);
    }
    else
    {
        recv_share(x_share);
    }

    Parameters<uint64_t> parameters;
    parameters.init_all();

    if (quick_verify)
    {
        if (party_id == 0)
        {
            std::cout << "[LLM2PC][QUICK] llama2-7b config/load check passed" << std::endl;
        }
        Party2PC::getInstance().sync();
        return;
    }

    benchmark_single_layer_and_estimate_full(full_llama2_config.num_layers,
                                             llm_loop,
                                             llm_warmup_rounds,
                                             x_share,
                                             parameters,
                                             (*llama2));
}
