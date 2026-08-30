#include "party2pc.h"
#include "twopc_protocols.h"
#include "layers.h"
#include "llmConfig.h"
#include "params.h"
#include "test_base.h"

static bool print_res = false;
static int loop = print_res ? 1 : 5;
static int warmup_rounds = print_res ? 0 : 3;

static void alexnet_test(int party_id);

static void resnet50_test(int party_id);

void CNN2pcTest::main_test(int party_id)
{
    alexnet_test(party_id);

    resnet50_test(party_id);
}

template <typename T>
static void share_data(twopc_protocols::ShareTensor<T> &x)
{
    static constexpr auto scale_bit = kFloat_Precision<T>;
    x.share.rand(-5, 5);

    Tensor<T>::mul(x.share, x.share, 1 << scale_bit);
}

template <typename T>
static void recv_share(twopc_protocols::ShareTensor<T> &x)
{
    x.zeros();
}

template <typename T>
static void nn_forward(twopc_protocols::ShareTensor<T> &x, Parameters<T> &parameters, Layers<T> &layer)
{
    layer(x, parameters);
    twopc_protocols::macCheckSimulate<T>(MAC_SIZE);
}

void alexnet_test(int party_id)
{
    printf("[Party %d] Starting AlexNet test\n", party_id);
    uint32_t batch_size = 128, channel = 3, height = 33, width = 33;  // shape of padded CIFAR 10
    Party2PC::getInstance().load_param<cnnRingType>();
    Layers<cnnRingType> *net = new SecAlexNet<cnnRingType>();

    std::vector<uint32_t> x_shape = {batch_size, channel, height, width};

    omp_set_num_threads(16);
    twopc_protocols::ShareTensor<cnnRingType> x_share(x_shape);
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

    test_func<Party2PC>(nn_forward<cnnRingType>, "AlexNet", loop, warmup_rounds, x_share, parameters, (*net));
}

void resnet50_test(int party_id)
{
    printf("[Party %d] Starting ResNet50 test\n", party_id);
    uint32_t batch_size = 1, channel = 3, height = 224, width = 224;  // shape of imageNet
    Party2PC::getInstance().load_param<cnnRingType>();
    Layers<cnnRingType> *net = new SecResNet50<cnnRingType>();

    std::vector<uint32_t> x_shape = {batch_size, channel, height, width};

    omp_set_num_threads(16);
    twopc_protocols::ShareTensor<cnnRingType> x_share(x_shape);
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

    test_func<Party2PC>(nn_forward<cnnRingType>, "ResNet50", loop, warmup_rounds, x_share, parameters, (*net));
}
