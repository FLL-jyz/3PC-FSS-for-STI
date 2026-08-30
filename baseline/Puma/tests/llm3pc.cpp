#include "party3pc.h"
#include "rss_protocols.h"
#include "layers.h"
#include "llmConfig.h"
#include "params.h"
#include "test_base.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <string>

static uint32_t batch_size = 1;
static bool print_res = false;
static int loop = 1;
static int warmup_rounds = 0;

static void tf_test(int party_id);

static void tf_test_64(int party_id);


static void bert_test(int party_id);

static void gpt2_test(int party_id);

static void gpt2_xl_test(int party_id);

static void llama2_test(int party_id);

static std::string selected_model()
{
    const char *value = std::getenv("LLM3PC_MODEL");
    return value == nullptr ? std::string() : std::string(value);
}

static bool should_run_model(const char *model_name)
{
    const std::string selected = selected_model();
    return selected.empty() || selected == model_name;
}

struct BenchmarkStats
{
    double wall_clock_seconds;
    double cpu_seconds;
    uint64_t sent_rounds;
    uint64_t recv_rounds;
    uint64_t sent_bytes;
    uint64_t recv_bytes;
};

static TransformerConfig make_single_layer_llama2_config(int seq_len)
{
    TransformerConfig config("llama2-7b", seq_len);
    config.num_layers = 1;
    return config;
}

static uint32_t make_reduced_num_layers(uint32_t full_num_layers)
{
    return std::max<uint32_t>(1, full_num_layers / 6);
}

template <typename F>
static BenchmarkStats benchmark_call(int benchmark_loop, int warmup_loop, F &&func)
{
    for (int i = 0; i < warmup_loop; ++i)
    {
        func();
    }

    Party3PC &party = Party3PC::getInstance();
    const uint64_t rounds_sent_before = party.rounds_sent();
    const uint64_t rounds_recv_before = party.rounds_received();
    const uint64_t bytes_sent_before = party.bytes_sent();
    const uint64_t bytes_recv_before = party.bytes_received();

    const auto wall_begin = std::chrono::steady_clock::now();
    const clock_t cpu_begin = clock();

    for (int i = 0; i < benchmark_loop; ++i)
    {
        func();
    }

    const clock_t cpu_end = clock();
    const auto wall_end = std::chrono::steady_clock::now();

    BenchmarkStats stats{};
    stats.wall_clock_seconds = std::chrono::duration<double>(wall_end - wall_begin).count() / benchmark_loop;
    stats.cpu_seconds = static_cast<double>(cpu_end - cpu_begin) / CLOCKS_PER_SEC / benchmark_loop;
    stats.sent_rounds = (party.rounds_sent() - rounds_sent_before) / benchmark_loop;
    stats.recv_rounds = (party.rounds_received() - rounds_recv_before) / benchmark_loop;
    stats.sent_bytes = (party.bytes_sent() - bytes_sent_before) / benchmark_loop;
    stats.recv_bytes = (party.bytes_received() - bytes_recv_before) / benchmark_loop;
    return stats;
}

static void print_scaled_benchmark(const std::string &test_name,
                                   uint32_t benchmark_num_layers,
                                   uint32_t full_num_layers,
                                   const BenchmarkStats &stats)
{
    if (Party3PC::getInstance().party_id != 0)
    {
        return;
    }

    const double scale = static_cast<double>(full_num_layers) / benchmark_num_layers;
    const uint64_t scaled_sent_rounds = static_cast<uint64_t>(std::llround(stats.sent_rounds * scale));
    const uint64_t scaled_recv_rounds = static_cast<uint64_t>(std::llround(stats.recv_rounds * scale));
    const uint64_t scaled_sent_bytes = static_cast<uint64_t>(std::llround(stats.sent_bytes * scale));
    const uint64_t scaled_recv_bytes = static_cast<uint64_t>(std::llround(stats.recv_bytes * scale));

    std::cout << "====================Test for " << test_name << " (" << benchmark_num_layers
              << "/" << full_num_layers << " layers)====================" << std::endl;
    std::cout << "Time:" << std::endl;
    std::cout << "Wall Clock time: " << stats.wall_clock_seconds << " sec\n";
    std::cout << "CPU time: " << stats.cpu_seconds << std::endl;
    std::cout << "Scaled to full model (" << full_num_layers << " layers):" << std::endl;
    std::cout << "Wall Clock time: " << stats.wall_clock_seconds * scale << " sec\n";
    std::cout << "CPU time: " << stats.cpu_seconds * scale << std::endl;

    std::cout << "Comm.: " << std::endl;
    std::cout << "\t Sent rounds:" << stats.sent_rounds << "\t Sent Bytes:"
              << convert_bytes_to_string(stats.sent_bytes) << std::endl;
    std::cout << "\t Recv rounds:" << stats.recv_rounds << "\t Recv Bytes:"
              << convert_bytes_to_string(stats.recv_bytes) << std::endl;

    std::cout << "Scaled to full model (" << full_num_layers << " layers):" << std::endl;
    std::cout << "\t Sent rounds:" << scaled_sent_rounds << "\t Sent Bytes:"
              << convert_bytes_to_string(scaled_sent_bytes) << std::endl;
    std::cout << "\t Recv rounds:" << scaled_recv_rounds << "\t Recv Bytes:"
              << convert_bytes_to_string(scaled_recv_bytes) << std::endl;
}

void LLM3pcTest::main_test(int party_id)
{   
    // bert_test(party_id);
    // Party3PC::getInstance().sync();
    // gpt2_test(party_id);
    // Party3PC::getInstance().sync();
    // llama2_test(party_id);
    // Party3PC::getInstance().sync();
    // tf_test(party_id);
    // Party3PC::getInstance().sync();
    // convert_test_sim(party_id);
    // Party3PC::getInstance().sync();
    // tf_test_64(party_id);
    // Party3PC::getInstance().sync();
    if (should_run_model("bert"))
    {
        bert_test(party_id);
        Party3PC::getInstance().sync();
    }
    if (should_run_model("gpt2"))
    {
        gpt2_test(party_id);
        Party3PC::getInstance().sync();
    }
    if (should_run_model("llama2") || should_run_model("llama2-7b"))
    {
        llama2_test(party_id);
        Party3PC::getInstance().sync();
    }
    // gpt2_xl_test(party_id);
}

static void share_data(RSSTensor<uint64_t> &x_share)
{
    Tensor<uint64_t> x(x_share.shape);
    x.rand(-5, 5);

    Tensor<uint64_t>::mul(x, x, 1 << FLOAT_PRECISION_64);

    rss_protocols::share(x, x_share);
}

static void recv_share(RSSTensor<uint64_t> &x_share)
{
    rss_protocols::recv_shares_from(0, x_share);
}

static void mixed_ffn_forward(RSSTensor<uint64_t> &x, Parameters<uint16_t> &parameters, Layers<uint64_t, uint16_t> &layer)
{
    layer(x, parameters);
}

static void mixed_layer_forward(RSSTensor<uint64_t> &x, Parameters<uint32_t> &parameters, Layers<uint64_t, uint32_t> &layer)
{
    layer(x, parameters);
}

static void mixed_llm_forward(RSSTensor<uint64_t> &x, Parameters<uint32_t> &parameters_u, Parameters<uint16_t> &parameters_v, Layers<uint64_t, uint32_t, uint16_t> &layer)
{
    layer(x, parameters_u, parameters_v);
    rss_protocols::macCheckSimulate<uint32_t>(MAC_SIZE);
}

static void mixed_llm_forward_ln(RSSTensor<uint64_t> &x, Parameters<uint32_t> &parameters_u, Parameters<uint16_t> &parameters_v,
                                 Layers<uint64_t, uint32_t, uint16_t> &layer, Layers<uint64_t, uint32_t> &layer_norm)
{
    layer_norm(x, parameters_u);
    layer(x, parameters_u, parameters_v);
    rss_protocols::macCheckSimulate<uint32_t>(MAC_SIZE);
}

static void mixed_gpt2_forward(RSSTensor<uint64_t> &x, Parameters<uint64_t> &parameters_t, Parameters<uint32_t> &parameters_u,
                               Parameters<uint16_t> &parameters_v, Layers<uint64_t, uint32_t, uint16_t> &layer)
{
    layer(x, parameters_t, parameters_u, parameters_v);
    rss_protocols::macCheckSimulate<uint32_t>(MAC_SIZE);
}

static void layer_forward(RSSTensor<uint64_t> &x, Parameters<uint64_t> &parameters, Layers<uint64_t> &layer)
{
    layer(x, parameters);
}

static void llm_forward(RSSTensor<uint64_t> &x, Parameters<uint64_t> &parameters, Layers<uint64_t> &layer)
{
    layer(x, parameters);
    rss_protocols::macCheckSimulate<uint64_t>(MAC_SIZE);
}

static void benchmark_single_layer_and_estimate_full(int full_num_layers,
                                                     int benchmark_loop,
                                                     int warmup_loop,
                                                     RSSTensor<uint64_t> &x,
                                                     Parameters<uint64_t> &parameters,
                                                     Layers<uint64_t> &layer)
{
    BenchmarkStats stats = benchmark_call(benchmark_loop,
                                          warmup_loop,
                                          [&]()
                                          {
                                              llm_forward(x, parameters, layer);
                                          });
    print_scaled_benchmark("llama2-7b single-layer", 1, full_num_layers, stats);
}

void tf_test(int party_id)
{
    Party3PC::getInstance().load_param<uint64_t>();
    TransformerConfig tf_config("transformer", 64);
    Layers<uint64_t, uint32_t> *attention = new SecMixedAttention(tf_config);
    Layers<uint64_t, uint32_t> *masked_attention = new SecMixedAttention(tf_config, true);
    Layers<uint64_t, uint16_t> *ffn = new SecMixedFFN(tf_config);
    Layers<uint64_t, uint32_t> *layer_norm = new SecMixedLayerNorm(tf_config.hidden_size);
    Layers<uint64_t, uint32_t, uint16_t> *encoder = new SecMixedEncoder(tf_config);
    Layers<uint64_t, uint32_t, uint16_t> *decoder = new SecMixedDecoder(tf_config);

    std::vector<uint32_t> x_shape = {batch_size, tf_config.seq_len, tf_config.hidden_size};

    omp_set_num_threads(8);
    RSSTensor<uint64_t> x_share(x_shape);
    if (party_id == 0)
    {
        share_data(x_share);
    }
    else
    {
        recv_share(x_share);
    }

    Parameters<uint32_t> parameters_u;
    parameters_u.init_all();
    Parameters<uint16_t> parameters_v;
    parameters_v.init_pc_cmp();
    parameters_v.init_gelu();

    test_func<Party3PC>(mixed_layer_forward, "attention", loop, warmup_rounds, x_share, parameters_u, (*attention));
    test_func<Party3PC>(mixed_layer_forward, "masked attention", loop, warmup_rounds, x_share, parameters_u, (*masked_attention));
    test_func<Party3PC>(mixed_ffn_forward, "ffn", loop, warmup_rounds, x_share, parameters_v, (*ffn));
    test_func<Party3PC>(mixed_layer_forward, "layernorm", loop, warmup_rounds, x_share, parameters_u, (*layer_norm));
    rss_protocols::macCheckSimulate<uint32_t>(MAC_SIZE);
    test_func<Party3PC>(mixed_llm_forward, "encoder", loop, warmup_rounds, x_share, parameters_u, parameters_v, (*encoder));
    test_func<Party3PC>(mixed_llm_forward, "decoder", loop, warmup_rounds, x_share, parameters_u, parameters_v, (*decoder));

    delete attention;
    delete masked_attention;
    delete ffn;
    delete layer_norm;
    delete encoder;
    delete decoder;
}

void tf_test_64(int party_id)
{
    Party3PC::getInstance().load_param<uint64_t>();
    TransformerConfig tf_config("transformer", 64);
    Layers<uint64_t> *attention = new SecAttention<uint64_t>(tf_config);
    Layers<uint64_t> *masked_attention = new SecAttention<uint64_t>(tf_config, true);
    Layers<uint64_t> *ffn = new SecFFN<uint64_t>(tf_config);
    Layers<uint64_t> *layer_norm = new SecLayerNorm<uint64_t>(tf_config.hidden_size);
    Layers<uint64_t> *encoder = new SecEncoder<uint64_t>(tf_config);
    Layers<uint64_t> *decoder = new SecDecoder<uint64_t>(tf_config);

    std::vector<uint32_t> x_shape = {batch_size, tf_config.seq_len, tf_config.hidden_size};

    omp_set_num_threads(16);
    RSSTensor<uint64_t> x_share(x_shape);
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

    test_func<Party3PC>(layer_forward, "attention 64", loop, warmup_rounds, x_share, parameters, (*attention));
    test_func<Party3PC>(layer_forward, "masked attention 64", loop, warmup_rounds, x_share, parameters, (*masked_attention));
    test_func<Party3PC>(layer_forward, "ffn 64", loop, warmup_rounds, x_share, parameters, (*ffn));
    test_func<Party3PC>(layer_forward, "layernorm 64", loop, warmup_rounds, x_share, parameters, (*layer_norm));
    rss_protocols::macCheckSimulate<uint64_t>(MAC_SIZE);
    test_func<Party3PC>(llm_forward, "encoder 64", loop, warmup_rounds, x_share, parameters, (*encoder));
    test_func<Party3PC>(llm_forward, "decoder 64", loop, warmup_rounds, x_share, parameters, (*decoder));

    delete attention;
    delete masked_attention;
    delete ffn;
    delete layer_norm;
    delete encoder;
    delete decoder;
}

void bert_test(int party_id)
{
    Party3PC::getInstance().load_param<uint64_t>();
    TransformerConfig full_bert_config("bert-base", 128);
    TransformerConfig bert_config = full_bert_config;
    bert_config.num_layers = make_reduced_num_layers(full_bert_config.num_layers);
    Layers<uint64_t, uint32_t, uint16_t> *bert = new SecMixedEncoder(bert_config);

    std::vector<uint32_t> x_shape = {batch_size, bert_config.seq_len, bert_config.hidden_size};

    omp_set_num_threads(8);
    RSSTensor<uint64_t> x_share(x_shape);
    if (party_id == 0)
    {   
        share_data(x_share);
    }
    else
    {   
        recv_share(x_share);
    }

    Parameters<uint32_t> parameters_u;
    parameters_u.init_all();
    Parameters<uint16_t> parameters_v;
    parameters_v.init_pc_cmp();
    parameters_v.init_gelu();

    Layers<uint64_t, uint32_t> *layer_norm = new SecMixedLayerNorm(bert_config.hidden_size);
    BenchmarkStats stats = benchmark_call(loop,
                                          warmup_rounds,
                                          [&]()
                                          {
                                              mixed_llm_forward_ln(x_share, parameters_u, parameters_v, (*bert), (*layer_norm));
                                          });
    print_scaled_benchmark("bert-base", bert_config.num_layers, full_bert_config.num_layers, stats);

    delete bert;
    delete layer_norm;
}

void gpt2_test(int party_id)
{
    Party3PC::getInstance().load_param<uint64_t>();
    TransformerConfig full_gpt2_config("gpt2", 64);
    TransformerConfig gpt2_config = full_gpt2_config;
    gpt2_config.num_layers = make_reduced_num_layers(full_gpt2_config.num_layers);
    Layers<uint64_t, uint32_t, uint16_t> *gpt2 = new SecMixedGPT2(gpt2_config);

    std::vector<uint32_t> x_shape = {batch_size, gpt2_config.seq_len, gpt2_config.hidden_size};

    omp_set_num_threads(8);
    RSSTensor<uint64_t> x_share(x_shape);
    if (party_id == 0)
    {
        share_data(x_share);
    }
    else
    {
        recv_share(x_share);
    }

    Parameters<uint64_t> parameters_t;
    parameters_t.init_all();
    Parameters<uint32_t> parameters_u;
    parameters_u.init_all();
    Parameters<uint16_t> parameters_v;
    parameters_v.init_pc_cmp();
    parameters_v.init_gelu();

    BenchmarkStats stats = benchmark_call(loop,
                                          warmup_rounds,
                                          [&]()
                                          {
                                              mixed_gpt2_forward(x_share, parameters_t, parameters_u, parameters_v, (*gpt2));
                                          });
    print_scaled_benchmark("gpt2", gpt2_config.num_layers, full_gpt2_config.num_layers, stats);

    delete gpt2;
}

void gpt2_xl_test(int party_id)
{
    Party3PC::getInstance().load_param<uint64_t>();
    TransformerConfig gpt2xl_config("gpt2-XL", 128);
    Layers<uint64_t, uint32_t, uint16_t> *gpt2 = new SecMixedGPT2(gpt2xl_config);

    std::vector<uint32_t> x_shape = {batch_size, gpt2xl_config.seq_len, gpt2xl_config.hidden_size};

    omp_set_num_threads(1);
    RSSTensor<uint64_t> x_share(x_shape);
    if (party_id == 0)
    {   
        share_data(x_share);
    }
    else
    {   
        recv_share(x_share);
    }

    Parameters<uint64_t> parameters_t;
    parameters_t.init_all();
    Parameters<uint32_t> parameters_u;
    parameters_u.init_all();
    Parameters<uint16_t> parameters_v;
    parameters_v.init_pc_cmp();
    parameters_v.init_gelu();

    test_func<Party3PC>(mixed_gpt2_forward, "gpt2-xl", loop, warmup_rounds, x_share, parameters_t, parameters_u, parameters_v, (*gpt2));

    delete gpt2;
}

void llama2_test(int party_id)
{
    Party3PC::getInstance().load_param<uint64_t>();
    TransformerConfig full_llama2_config("llama2-7b", 16);
    TransformerConfig llama2_config = make_single_layer_llama2_config(16);
    omp_set_num_threads(8);
    Layers<uint64_t> *llama2 = new SecLlama2<uint64_t>(llama2_config);

    std::vector<uint32_t> x_shape = {batch_size, llama2_config.seq_len, llama2_config.hidden_size};

    RSSTensor<uint64_t> x_share(x_shape);
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

    benchmark_single_layer_and_estimate_full(static_cast<int>(full_llama2_config.num_layers),
                                             loop,
                                             warmup_rounds,
                                             x_share,
                                             parameters,
                                             (*llama2));

    delete llama2;
}
