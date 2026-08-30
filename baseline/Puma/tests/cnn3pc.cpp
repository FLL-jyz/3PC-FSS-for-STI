#include "party3pc.h"
#include "rss_protocols.h"
#include "layers.h"
#include "llmConfig.h"
#include "params.h"
#include "test_base.h"
#include <chrono>
#include <cmath>
#include <string>

static bool print_res = false;
static int loop = 1;
static int warmup_rounds = 0;
static int num_threads = 8;

static void alexnet_test(int party_id);

static void resnet50_test(int party_id);

struct BenchmarkStats
{
    double wall_clock_seconds = 0;
    double cpu_seconds = 0;
    uint64_t sent_rounds = 0;
    uint64_t recv_rounds = 0;
    uint64_t sent_bytes = 0;
    uint64_t recv_bytes = 0;
};

void CNN3pcTest::main_test(int party_id)
{
    alexnet_test(party_id);
    Party3PC::getInstance().sync();

    resnet50_test(party_id);
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

static void accumulate_stats(BenchmarkStats &total, const BenchmarkStats &stats, uint32_t scale = 1)
{
    total.wall_clock_seconds += stats.wall_clock_seconds * scale;
    total.cpu_seconds += stats.cpu_seconds * scale;
    total.sent_rounds += stats.sent_rounds * scale;
    total.recv_rounds += stats.recv_rounds * scale;
    total.sent_bytes += stats.sent_bytes * scale;
    total.recv_bytes += stats.recv_bytes * scale;
}

static void print_component_benchmark(const std::string &name, const BenchmarkStats &stats, uint32_t scale = 1)
{
    if (Party3PC::getInstance().party_id != 0)
    {
        return;
    }

    std::cout << "---- " << name;
    if (scale > 1)
    {
        std::cout << " x" << scale;
    }
    std::cout << std::endl;
    std::cout << "Wall Clock time: " << stats.wall_clock_seconds;
    if (scale > 1)
    {
        std::cout << " sec (scaled " << stats.wall_clock_seconds * scale << " sec)";
    }
    else
    {
        std::cout << " sec";
    }
    std::cout << std::endl;
    std::cout << "CPU time: " << stats.cpu_seconds;
    if (scale > 1)
    {
        std::cout << " (scaled " << stats.cpu_seconds * scale << ")";
    }
    std::cout << std::endl;
    std::cout << "Sent rounds: " << stats.sent_rounds;
    if (scale > 1)
    {
        std::cout << " (scaled " << stats.sent_rounds * scale << ")";
    }
    std::cout << "\tSent Bytes: " << convert_bytes_to_string(stats.sent_bytes);
    if (scale > 1)
    {
        std::cout << " (scaled " << convert_bytes_to_string(stats.sent_bytes * scale) << ")";
    }
    std::cout << std::endl;
    std::cout << "Recv rounds: " << stats.recv_rounds;
    if (scale > 1)
    {
        std::cout << " (scaled " << stats.recv_rounds * scale << ")";
    }
    std::cout << "\tRecv Bytes: " << convert_bytes_to_string(stats.recv_bytes);
    if (scale > 1)
    {
        std::cout << " (scaled " << convert_bytes_to_string(stats.recv_bytes * scale) << ")";
    }
    std::cout << std::endl;
}

static void print_total_benchmark(const std::string &test_name, const std::string &mode, const BenchmarkStats &stats)
{
    if (Party3PC::getInstance().party_id != 0)
    {
        return;
    }

    std::cout << "====================Test for " << test_name << " (" << mode << ")====================" << std::endl;
    std::cout << "Time:" << std::endl;
    std::cout << "Wall Clock time: " << stats.wall_clock_seconds << " sec\n";
    std::cout << "CPU time: " << stats.cpu_seconds << std::endl;
    std::cout << "Comm.: " << std::endl;
    std::cout << "\t Sent rounds:" << stats.sent_rounds << "\t Sent Bytes:"
              << convert_bytes_to_string(stats.sent_bytes) << std::endl;
    std::cout << "\t Recv rounds:" << stats.recv_rounds << "\t Recv Bytes:"
              << convert_bytes_to_string(stats.recv_bytes) << std::endl;
}

template <typename T, typename F>
static RSSTensor<T> benchmark_layer(const std::string &name,
                                    BenchmarkStats &total,
                                    F &&forward,
                                    uint32_t scale = 1)
{
    RSSTensor<T> output;
    BenchmarkStats stats = benchmark_call(loop, warmup_rounds, [&]() {
        output = forward();
    });
    print_component_benchmark(name, stats, scale);
    accumulate_stats(total, stats, scale);
    Party3PC::getInstance().sync();
    return output;
}

template <typename T>
static void benchmark_mac_check(const std::string &name, BenchmarkStats &total)
{
    BenchmarkStats stats = benchmark_call(loop, warmup_rounds, [&]() {
        rss_protocols::macCheckSimulate<T>(MAC_SIZE);
    });
    print_component_benchmark(name, stats);
    accumulate_stats(total, stats);
    Party3PC::getInstance().sync();
}

template <typename T>
static RSSTensor<T> benchmark_resnet_blocks(const std::string &stage_name,
                                            std::vector<BottleNeck<T>> &blocks,
                                            RSSTensor<T> &input,
                                            Parameters<T> &parameters,
                                            BenchmarkStats &total)
{
    always_assert(!blocks.empty());

    RSSTensor<T> current = input;
    for (size_t block_index = 0; block_index < blocks.size(); ++block_index)
    {
        auto &block = blocks[block_index];
        current = benchmark_layer<T>(stage_name + ".block" + std::to_string(block_index), total, [&]() {
            return block(current, parameters);
        });
    }

    return current;
}

template <typename T>
static void share_data(RSSTensor<T> &x_share)
{
    static constexpr auto scale_bit = kFloat_Precision<T>;
    Tensor<T> x(x_share.shape);
    x.rand(-5, 5);

    Tensor<T>::mul(x, x, 1 << scale_bit);

    rss_protocols::share(x, x_share);
}

template <typename T>
static void recv_share(RSSTensor<T> &x_share)
{
    rss_protocols::recv_shares_from(0, x_share);
}

void alexnet_test(int party_id)
{
    printf("[Party %d] Starting AlexNet test\n", party_id);
    uint32_t batch_size = 128, channel = 3, height = 33, width = 33;  // shape of padded CIFAR 10
    Party3PC::getInstance().load_param<cnnRingType>();
    SecAlexNet<cnnRingType> net;

    std::vector<uint32_t> x_shape = {batch_size, channel, height, width};

    omp_set_num_threads(num_threads);
    RSSTensor<cnnRingType> x_share(x_shape);
    if (party_id == 0)
    {
        share_data(x_share);
    }
    else
    {
        recv_share(x_share);
    }

    Parameters<cnnRingType> parameters;
    parameters.init_all();

    BenchmarkStats total;
    RSSTensor<cnnRingType> current = x_share;

    current = benchmark_layer<cnnRingType>("AlexNet.conv1", total, [&]() { return (*net.conv1)(current); });
    current = benchmark_layer<cnnRingType>("AlexNet.pool1", total, [&]() { return (*net.pool1)(current, parameters); });
    current = benchmark_layer<cnnRingType>("AlexNet.relu1", total, [&]() { return (*net.relu1)(current, parameters); });

    current = benchmark_layer<cnnRingType>("AlexNet.conv2", total, [&]() { return (*net.conv2)(current); });
    current = benchmark_layer<cnnRingType>("AlexNet.pool2", total, [&]() { return (*net.pool2)(current, parameters); });
    current = benchmark_layer<cnnRingType>("AlexNet.relu2", total, [&]() { return (*net.relu2)(current, parameters); });

    current = benchmark_layer<cnnRingType>("AlexNet.conv3", total, [&]() { return (*net.conv3)(current); });
    current = benchmark_layer<cnnRingType>("AlexNet.relu3", total, [&]() { return (*net.relu3)(current, parameters); });

    current = benchmark_layer<cnnRingType>("AlexNet.conv4", total, [&]() { return (*net.conv4)(current); });
    current = benchmark_layer<cnnRingType>("AlexNet.relu4", total, [&]() { return (*net.relu4)(current, parameters); });

    current = benchmark_layer<cnnRingType>("AlexNet.conv5", total, [&]() { return (*net.conv5)(current); });
    current = benchmark_layer<cnnRingType>("AlexNet.relu5", total, [&]() { return (*net.relu5)(current, parameters); });

    current.reshape({current.size() / 256, 256});
    current = benchmark_layer<cnnRingType>("AlexNet.fc1", total, [&]() { return (*net.fc1)(current); });
    current = benchmark_layer<cnnRingType>("AlexNet.relu6", total, [&]() { return (*net.relu6)(current, parameters); });
    current = benchmark_layer<cnnRingType>("AlexNet.fc2", total, [&]() { return (*net.fc2)(current); });
    current = benchmark_layer<cnnRingType>("AlexNet.relu7", total, [&]() { return (*net.relu7)(current, parameters); });
    current = benchmark_layer<cnnRingType>("AlexNet.fc3", total, [&]() { return (*net.fc3)(current); });
    current = benchmark_layer<cnnRingType>("AlexNet.relu8", total, [&]() { return (*net.relu8)(current, parameters); });

    benchmark_mac_check<cnnRingType>("AlexNet.macCheck", total);
    print_total_benchmark("AlexNet", "layer-wise aggregated", total);
    open_print(current, party_id, print_res);
}

void resnet50_test(int party_id)
{
    printf("[Party %d] Starting ResNet50 test\n", party_id);
    uint32_t batch_size = 1, channel = 3, height = 224, width = 224;  // shape of imageNet
    Party3PC::getInstance().load_param<cnnRingType>();
    SecResNet50<cnnRingType> net;

    std::vector<uint32_t> x_shape = {batch_size, channel, height, width};

    omp_set_num_threads(num_threads);
    RSSTensor<cnnRingType> x_share(x_shape);
    if (party_id == 0)
    {
        share_data(x_share);
    }
    else
    {
        recv_share(x_share);
    }

    Parameters<cnnRingType> parameters;
    parameters.init_all();

    BenchmarkStats total;
    RSSTensor<cnnRingType> current = x_share;

    current = benchmark_layer<cnnRingType>("ResNet50.stem.conv", total, [&]() { return (*net.conv)(current); });
    current = benchmark_layer<cnnRingType>("ResNet50.stem.bn", total, [&]() { return (*net.bn)(current); });
    current = benchmark_layer<cnnRingType>("ResNet50.stem.relu", total, [&]() { return (*net.relu)(current, parameters); });
    current = benchmark_layer<cnnRingType>("ResNet50.stem.pool", total, [&]() { return (*net.pool)(current, parameters); });

    current = benchmark_resnet_blocks("ResNet50.conv2_x", net.conv2_x, current, parameters, total);
    current = benchmark_resnet_blocks("ResNet50.conv3_x", net.conv3_x, current, parameters, total);
    current = benchmark_resnet_blocks("ResNet50.conv4_x", net.conv4_x, current, parameters, total);
    current = benchmark_resnet_blocks("ResNet50.conv5_x", net.conv5_x, current, parameters, total);

    current = benchmark_layer<cnnRingType>("ResNet50.avg_pool", total, [&]() { return (*net.avg_pool)(current); });
    current.reshape({current.shape[0], current.size() / current.shape[0]});
    current = benchmark_layer<cnnRingType>("ResNet50.fc", total, [&]() { return (*net.fc)(current); });

    benchmark_mac_check<cnnRingType>("ResNet50.macCheck", total);
    print_total_benchmark("ResNet50", "block-wise aggregated", total);
    open_print(current, party_id, print_res);
}
