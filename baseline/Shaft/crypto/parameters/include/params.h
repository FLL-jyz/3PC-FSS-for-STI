#ifndef PARAMS_H
#define PARAMS_H
#include <array>
#include "globals.h"
#include "my_math.h"
#include "tensor.h"
#include "party2pc.h"
#include "2pcdpf.h"

// follow the pattern of debug level 2
template <typename T>
struct pc_cmp_param
{
    uint32_t eval_bitlength;
    T self_r0;
    T self_r1;
    T r_with_pre;
    T r_with_next;
    Tensor<T> r_with_pre_bits;
    Tensor<T> r_with_next_bits;
    std::vector<TwoPcDPFKeyPack> keys;
    T round1_r;
    T round2_r;
    Tensor<T> round1_real_table_with_pre;
    Tensor<T> round1_real_table_with_next;
    Tensor<T> round2_real_table_with_pre;
    Tensor<T> round2_real_table_with_next;
};

template <typename T>
class Parameters
{
public:
    struct pc_cmp_param<T> pc_cmp;
    std::vector<pc_cmp_param<T>> extra_pc_cmp;
    std::array<double, 8> gelu_fourier_beta;
    int party_id;

    static constexpr auto scale_bit = []
    {
        if constexpr (std::is_same_v<T, uint64_t>)
        {
            return FLOAT_PRECISION_64;
        }
        else if constexpr (std::is_same_v<T, uint32_t>)
        {
            return FLOAT_PRECISION_32;
        }
        else if constexpr (std::is_same_v<T, uint16_t>)
        {
            return FLOAT_PRECISION_16;
        }
        else
        {
            return 1;
        }
    }();

    Parameters()
    {
        this->party_id = Party2PC::getInstance().party_id;
        gelu_fourier_beta.fill(0.0);
    }

    static double gelu_delta_real(double x)
    {
        return static_cast<double>(GeLU(x)) - static_cast<double>(MY_RELU(x));
    }

    static double gelu_fourier_series(const std::array<double, 8> &beta, double x)
    {
        constexpr double kPi = 3.141592653589793238462643383279502884;
        constexpr double kDomain = 4.0;

        if (x <= 0.0 || x >= kDomain)
        {
            return 0.0;
        }

        double acc = 0.0;
        for (size_t term = 0; term < beta.size(); ++term)
        {
            const double harmonic = static_cast<double>(term + 1);
            acc += beta[term] * std::sin(harmonic * kPi * x / kDomain);
        }
        return acc;
    }

    void init_gelu_fourier_beta()
    {
        constexpr double kPi = 3.141592653589793238462643383279502884;
        constexpr double kDomain = 4.0;
        constexpr uint32_t kSimpsonSteps = 1U << 14;
        static_assert((kSimpsonSteps % 2) == 0, "Simpson integration requires an even number of steps");

        const double step = kDomain / static_cast<double>(kSimpsonSteps);
        for (size_t term = 0; term < gelu_fourier_beta.size(); ++term)
        {
            const double harmonic = static_cast<double>(term + 1);
            double weighted_sum = 0.0;
            for (uint32_t i = 0; i <= kSimpsonSteps; ++i)
            {
                const double x = step * static_cast<double>(i);
                const double basis = std::sin(harmonic * kPi * x / kDomain);
                const double sample = gelu_delta_real(x) * basis;
                const double weight = (i == 0 || i == kSimpsonSteps) ? 1.0 : ((i & 1U) ? 4.0 : 2.0);
                weighted_sum += weight * sample;
            }

            const double integral = (step / 3.0) * weighted_sum;
            gelu_fourier_beta[term] = (2.0 / kDomain) * integral;
        }
    }

    uint32_t resolve_pc_cmp_bitlength(uint32_t eval_bitlength) const
    {
        if (eval_bitlength != 0)
        {
            return eval_bitlength;
        }
        return static_cast<uint32_t>(int(log2(2 * sizeof(T) * 8)) + 1);
    }

    void init_pc_cmp_variant(pc_cmp_param<T> &variant, uint32_t eval_bitlength)
    {
        variant.eval_bitlength = eval_bitlength;
        variant.self_r0 = 0;
        variant.self_r1 = 0;

        variant.r_with_pre = 0;
        variant.r_with_next = 0;

        variant.r_with_pre_bits.allocate({eval_bitlength});
        variant.r_with_next_bits.allocate({eval_bitlength});

        variant.r_with_pre_bits.zero();
        variant.r_with_next_bits.fill(1);
        variant.r_with_next_bits.data[0] = 0;

        variant.round1_r = 0;
        variant.round2_r = 0;

        uint32_t double_bit_length = 2 * eval_bitlength;
        variant.round1_real_table_with_pre.allocate({double_bit_length});
        variant.round1_real_table_with_next.allocate({double_bit_length});
        variant.round2_real_table_with_pre.allocate({double_bit_length});
        variant.round2_real_table_with_next.allocate({double_bit_length});

        variant.round1_real_table_with_pre.zero();
        variant.round1_real_table_with_next.zero();
        variant.round2_real_table_with_pre.zero();
        variant.round2_real_table_with_next.zero();

        variant.round1_real_table_with_pre.data[0] = 1;
        variant.round2_real_table_with_pre.data[0] = 1;

        variant.keys.resize(1);
        GroupElement payload(0, eval_bitlength);
        size_t j = rand() % (2 * eval_bitlength);
        GroupElement idx(j, eval_bitlength);
        variant.keys[0] = keyGenTwoPcDPF(eval_bitlength, eval_bitlength, idx, payload, GenParams(false)).first;
    }

    void init_extra_pc_cmp_variant(uint32_t eval_bitlength)
    {
        if (eval_bitlength == pc_cmp.eval_bitlength)
        {
            return;
        }

        for (auto &variant : extra_pc_cmp)
        {
            if (variant.eval_bitlength == eval_bitlength)
            {
                return;
            }
        }

        extra_pc_cmp.emplace_back();
        init_pc_cmp_variant(extra_pc_cmp.back(), eval_bitlength);
    }

    const pc_cmp_param<T> &select_pc_cmp(uint32_t eval_bitlength)
    {
        const uint32_t resolved_bitlength = resolve_pc_cmp_bitlength(eval_bitlength);
        if (resolved_bitlength == pc_cmp.eval_bitlength)
        {
            return pc_cmp;
        }

        for (auto &variant : extra_pc_cmp)
        {
            if (variant.eval_bitlength == resolved_bitlength)
            {
                return variant;
            }
        }

        extra_pc_cmp.emplace_back();
        init_pc_cmp_variant(extra_pc_cmp.back(), resolved_bitlength);
        return extra_pc_cmp.back();
    }

    void init_pc_cmp()
    {
        init_pc_cmp_variant(pc_cmp, resolve_pc_cmp_bitlength(0));
        extra_pc_cmp.clear();

        const uint32_t type_bitlength = static_cast<uint32_t>(sizeof(T) * 8);
        if (type_bitlength >= 47)
        {
            init_extra_pc_cmp_variant(47);
        }
        if (type_bitlength >= 10)
        {
            init_extra_pc_cmp_variant(10);
        }
        if (type_bitlength >= 8)
        {
            init_extra_pc_cmp_variant(8);
        }
        init_extra_pc_cmp_variant(type_bitlength);
    }

    void init_gelu()
    {
        init_gelu_fourier_beta();
    }

    void init_all()
    {   
        init_pc_cmp();
        init_gelu();
    }
};

#endif // PARAMS_H