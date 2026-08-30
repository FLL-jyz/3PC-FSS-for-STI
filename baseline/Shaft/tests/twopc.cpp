#include <functional>
#include "party2pc.h"
#include "twopc_protocols.h"
#include "tensor.h"
#include "params.h"
#include "test_base.h"
#include "globals.h"

using namespace twopc_protocols;

static uint32_t num = 100;
static bool print_res = false;
static int loop = print_res ? 1 : 10;
static int warm_up = print_res ? 0 : 5;
static std::vector<uint32_t> x_shape = {num, num};
static std::vector<uint32_t> y_shape = {num, num};
static std::vector<uint32_t> z_shape = {num, num};

template <typename T>
void add_test(ShareTensor<T> &, ShareTensor<T> &, ShareTensor<T> &);

template <typename T>
void mul_test(ShareTensor<T> &, ShareTensor<T> &, ShareTensor<T> &);

template <typename T>
void matmul_test(ShareTensor<T> &, ShareTensor<T> &, ShareTensor<T> &);

template <typename T>
void ge_test(ShareTensor<T> &, ShareTensor<T> &, ShareTensor<T> &, Parameters<T> &);

template <typename T>
void rsqrt_test(ShareTensor<T> &, ShareTensor<T> &, Parameters<T> &);

template <typename T>
void softmax_test(ShareTensor<T> &, ShareTensor<T> &, Parameters<T> &);

template <typename T>
void drelu_test(ShareTensor<T> &, ShareTensor<T> &, Parameters<T> &);

template <typename T>
void relu_test(ShareTensor<T> &, ShareTensor<T> &, Parameters<T> &);

template <typename T>
void gelu_test(ShareTensor<T> &, ShareTensor<T> &, Parameters<T> &);

template <typename T>
void nexp_test(ShareTensor<T> &, ShareTensor<T> &, Parameters<T> &);

template <typename T>
void max_test(ShareTensor<T> &, ShareTensor<T> &, Parameters<T> &);

template <typename T>
void convert_test(ShareTensor<T> &, int party_id);

template <typename T>
static void share_data(Tensor<T> &x, Tensor<T> &y, ShareTensor<T> &x_share, ShareTensor<T> &y_share)
{
    // Share Tensor inputs into the secret-share domain used by the protocol core.
    share(x, x_share);
    share(y, y_share);
}

template <typename T>
static void recv_share(ShareTensor<T> &x_share, ShareTensor<T> &y_share)
{
    recv_shares_from(0, x_share);
    recv_shares_from(0, y_share);
}

void TwoPcProtocolTest::main_test(int party_id)
{
    omp_set_num_threads(16);

    Tensor<ringType> x_input(x_shape), y_input(y_shape), z_output(z_shape), max_output({num});
    if (party_id == 0)
    {
        x_input.rand(-5, 5);
        y_input.rand(0, 5);
        Tensor<ringType>::mul(x_input, x_input, 1 << kFloat_Precision<ringType>);
        Tensor<ringType>::mul(y_input, y_input, 1 << kFloat_Precision<ringType>);
    }

    ShareTensor<ringType> x_share(x_shape), y_share(y_shape);
    if (party_id == 0)
    {
        share_data(x_input, y_input, x_share, y_share);
    }
    else
    {
        recv_share(x_share, y_share);
    }

    Parameters<ringType> parameters;
    parameters.init_all();

    ShareTensor<ringType> z_share(z_shape);

    // test addition
    test_func<Party2PC>(add_test<ringType>, "addition", loop, warm_up, x_share, y_share, z_share);
    restore(z_share, z_output);
    if (print_res && party_id == 0) z_output.print();

    // test multiplication
    test_func<Party2PC>(mul_test<ringType>, "multiplication", loop, warm_up, x_share, y_share, z_share);
    restore(z_share, z_output);
    if (print_res && party_id == 0) z_output.print();

    // test matMul
    test_func<Party2PC>(matmul_test<ringType>, "matmul", loop, warm_up, x_share, y_share, z_share);
    restore(z_share, z_output);
    if (print_res && party_id == 0) z_output.print();

    // test greaterEqual
    test_func<Party2PC>(ge_test<ringType>, "greater equal", loop, warm_up, x_share, y_share, z_share, parameters);
    restore(z_share, z_output);
    if (print_res && party_id == 0) z_output.print();

    // test rsqrt
    test_func<Party2PC>(rsqrt_test<ringType>, "rsqrt", loop, warm_up, x_share, z_share, parameters);
    restore(z_share, z_output);
    if (print_res && party_id == 0) z_output.print();

    // test softmax
    test_func<Party2PC>(softmax_test<ringType>, "softmax", loop, warm_up, x_share, z_share, parameters);
    restore(z_share, z_output);
    if (print_res && party_id == 0) z_output.print();

    // test relu
    test_func<Party2PC>(relu_test<ringType>, "relu", loop, warm_up, x_share, z_share, parameters);
    restore(z_share, z_output);
    if (print_res && party_id == 0) z_output.print();

    // test gelu
    test_func<Party2PC>(gelu_test<ringType>, "gelu", loop, warm_up, x_share, z_share, parameters);
    restore(z_share, z_output);
    if (print_res && party_id == 0) z_output.print();

    // test nexp
    test_func<Party2PC>(nexp_test<ringType>, "nexp", loop, warm_up, x_share, z_share, parameters);
    restore(z_share, z_output);
    if (print_res && party_id == 0) z_output.print();

    // test max
    ShareTensor<ringType> max_share({num});
    test_func<Party2PC>(max_test<ringType>, "max", loop, warm_up, x_share, max_share, parameters);
    restore(max_share, max_output);
    if (print_res && party_id == 0) max_output.print();

    // test type conversion
    test_func<Party2PC>(convert_test<ringType>, "type conversion", loop, warm_up, x_share, party_id);
}

void TwoPcProtocolBenchmarkTest::main_test(int party_id)
{
    omp_set_num_threads(16);
    for (uint32_t num = 1000; num < 1000000; num *= 10)
    {
        ShareTensor<ringType> x_share({num});
        if (party_id == 0)
        {
            Tensor<ringType> x({num});
            x.rand(-5, 5);
            Tensor<ringType>::mul(x, x, 1 << kFloat_Precision<ringType>);
            share(x, x_share);
        }
        else
        {
            recv_shares_from(0, x_share);
        }

        Parameters<ringType> parameters;
        parameters.init_all();

        ShareTensor<ringType> z_share({num});

        if (party_id == 0)
        {
            std::cout << "********************** num = " << num << " **********************" << std::endl;
        }
        // test softmax
        test_func<Party2PC>(softmax_test<ringType>, "softmax", loop, warm_up, x_share, z_share, parameters);
        // // test drelu
        // test_func<Party2PC>(drelu_test<ringType>, "drelu", loop, warm_up, x_share, x_share, parameters);

        // // test relu
        // test_func<Party2PC>(relu_test<ringType>, "relu", loop, warm_up, x_share, z_share, parameters);

        // test gelu
        test_func<Party2PC>(gelu_test<ringType>, "gelu", loop, warm_up, x_share, z_share, parameters);

        // test rsqrt
        test_func<Party2PC>(rsqrt_test<ringType>, "rsqrt", loop, warm_up, x_share, z_share, parameters);

    }
}

template <typename T>
void add_test(ShareTensor<T> &x_share, ShareTensor<T> &y_share, ShareTensor<T> &z_share)
{
    add(x_share, y_share, z_share);
}

template <typename T>
void mul_test(ShareTensor<T> &x_share, ShareTensor<T> &y_share, ShareTensor<T> &z_share)
{
    mul(x_share, y_share, z_share, true, IS_MALICIOUS);
}

template <typename T>
void matmul_test(ShareTensor<T> &x_share, ShareTensor<T> &y_share, ShareTensor<T> &z_share)
{
    matMul(x_share, y_share, z_share, true, IS_MALICIOUS);
}

template <typename T>
void ge_test(ShareTensor<T> &x_share, ShareTensor<T> &y_share, ShareTensor<T> &z_share, Parameters<T> &param)
{
    greaterEqual(x_share, y_share, z_share, param, true, IS_MALICIOUS);
    macCheckSimulate<T>(MAC_SIZE);
}

template <typename T>
void rsqrt_test(ShareTensor<T> &x_share, ShareTensor<T> &z_share, Parameters<T> &param)
{
    rsqrt(x_share, z_share, param, IS_MALICIOUS);
    macCheckSimulate<T>(MAC_SIZE);
}

template <typename T>
void softmax_test(ShareTensor<T> &x_share, ShareTensor<T> &z_share, Parameters<T> &param)
{
    softmax_forward(x_share, z_share, param, IS_MALICIOUS);
    macCheckSimulate<T>(MAC_SIZE);
}

template <typename T>
void drelu_test(ShareTensor<T> &x_share, ShareTensor<T> &z_share, Parameters<T> &param)
{
    nonNegative(x_share, z_share, param, true, IS_MALICIOUS);
    macCheckSimulate<T>(MAC_SIZE);
}

template <typename T>
void relu_test(ShareTensor<T> &x_share, ShareTensor<T> &z_share, Parameters<T> &param)
{
    select(x_share, x_share, z_share, param, IS_MALICIOUS);
    macCheckSimulate<T>(MAC_SIZE);
}

template <typename T>
void gelu_test(ShareTensor<T> &x_share, ShareTensor<T> &z_share, Parameters<T> &param)
{
    gelu(x_share, z_share, param, IS_MALICIOUS);
    macCheckSimulate<T>(MAC_SIZE);
}

template <typename T>
void nexp_test(ShareTensor<T> &x_share, ShareTensor<T> &z_share, Parameters<T> &param)
{
    neg_exp(x_share, z_share, param, IS_MALICIOUS);
    macCheckSimulate<T>(MAC_SIZE);
}

template <typename T>
void max_test(ShareTensor<T> &x_share, ShareTensor<T> &z_share, Parameters<T> &param)
{
    max_last_dim(x_share, z_share, param, IS_MALICIOUS);
    macCheckSimulate<T>(MAC_SIZE);
}
