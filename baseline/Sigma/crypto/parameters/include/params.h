#ifndef PARAMS_H
#define PARAMS_H
#include "globals.h"
#include "my_math.h"
#include "tensor.h"
#include "party2pc.h"
#include "fss.h"
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
class LUT_Param
{
public:
    Tensor<T> table;
    uint32_t table_size;
    T self_r0;
    T self_r1;
    T r_with_pre;
    T r_with_next;

    // Backward-compatible field used by existing LUT online code paths.
    Tensor<T> onehot_table;
    Tensor<T> onehot_table_with_pre;
    Tensor<T> onehot_table_with_next;

    void init(uint32_t table_size)
    {
        this->table_size = table_size;
        table.allocate({table_size});

        self_r0 = 0;
        self_r1 = 0;
        r_with_pre = 0;
        r_with_next = 0;

        onehot_table.allocate({table_size});
        onehot_table_with_pre.allocate({table_size});
        onehot_table_with_next.allocate({table_size});

        onehot_table.zero();
        onehot_table_with_pre.zero();
        onehot_table_with_next.zero();

        onehot_table.data[0] = 1;
        onehot_table_with_pre.data[0] = 1;
    }
};

template <typename T>
class Parameters
{
public:
    struct pc_cmp_param<T> pc_cmp;
    std::vector<pc_cmp_param<T>> extra_pc_cmp;
    LUT_Param<T> nexp2_param;
    LUT_Param<T> nexpb_param;
    LUT_Param<T> exp_hi_param;
    LUT_Param<T> exp_lo_param;
    LUT_Param<T> inv_param;
    LUT_Param<T> inv_chopped_param;
    uint32_t inv_chopped_max_k;
    LUT_Param<T> sqrt_nexpb_param;
    LUT_Param<T> rsqrt_shift_param;
    LUT_Param<T> rsqrt_param;
    LUT_Param<T> gelu_param;
    LUT_Param<T> swiglu_param;
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
        this->inv_chopped_max_k = 0;
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

    void init_nexp2()
    {
        uint32_t table_size = 2 * scale_bit + 1;
        nexp2_param.init(table_size);
        for (int i = 0; i < table_size; i++)
        {
            nexp2_param.table.data[i] = 1 << (scale_bit - i + scale_bit);
        }
    }

    void init_nexpb()
    {
        uint32_t table_size = (int)(log(pow(2, 2 * scale_bit)) / log(SCALE_BASE)) + 1;
        nexpb_param.init(table_size);
        for(int i = 0; i < table_size; i++)
        {
            nexpb_param.table.data[i] = (T)((pow(2, scale_bit) / pow(SCALE_BASE, i)) * (1 << scale_bit));
        }
    }

    void init_exp()
    {
        constexpr int exp_frac_bits = 12;
        const double scale = static_cast<double>(1ULL << exp_frac_bits);

        exp_hi_param.init(256);
        exp_lo_param.init(256);

        for (int i = 0; i < 256; ++i)
        {
            const int64_t v_hi = static_cast<int64_t>(std::floor(std::exp(-static_cast<double>(i) / 16.0) * scale));
            const int64_t v_lo = static_cast<int64_t>(std::floor(std::exp(-static_cast<double>(i) / 4096.0) * scale));
            exp_hi_param.table.data[i] = static_cast<T>(v_hi);
            exp_lo_param.table.data[i] = static_cast<T>(v_lo);
        }
    }

    void init_inv()
    {
        uint32_t table_size = (uint32_t)ceil((1.0  - 1.0 / SCALE_BASE) * (1 << scale_bit));  // 1 / b * (2 ^ f)
        float table_scale = 1 << scale_bit;
        inv_param.init(table_size);

        for(int i = 0; i < table_size; i++)
        {
            float real_value = i / table_scale + 1.0 / SCALE_BASE;
            inv_param.table.data[i] = (T)((1 / real_value) * (1 << scale_bit));
        }
    }

    void init_inv_chopped(uint32_t max_k)
    {
        inv_chopped_max_k = max_k;

        constexpr uint32_t chopped_frac_bits = 6;
        uint32_t int_bits = 0;
        while ((1U << int_bits) < (max_k + 1U))
        {
            ++int_bits;
        }
        const uint32_t q = chopped_frac_bits + int_bits;
        const uint32_t table_size = (1U << q);

        inv_chopped_param.init(table_size);
        const uint64_t numerator = (1ULL << (scale_bit + chopped_frac_bits));

        // Safety fallback; index 0 should not appear for z in [1, k].
        inv_chopped_param.table.data[0] = static_cast<T>(numerator);
        for (uint32_t i = 1; i < table_size; ++i)
        {
            inv_chopped_param.table.data[i] = static_cast<T>(numerator / static_cast<uint64_t>(i));
        }
    }

    void init_rsqrt()
    {
        constexpr uint32_t bit_width = static_cast<uint32_t>(sizeof(T) * 8);
        rsqrt_shift_param.init(bit_width + 1);
        // Input index k is bit-length in [1, bit_width], table returns 2^(bit_width-k).
        rsqrt_shift_param.table.data[0] = static_cast<T>(0);
        for (uint32_t k = 1; k <= bit_width; ++k)
        {
            rsqrt_shift_param.table.data[k] = static_cast<T>(1ULL << (bit_width - k));
        }

        // SIGMA-style custom floating encoding: e(6 bits) + m(7 bits) => 13-bit index.
        constexpr uint32_t exponent_bits = 6;
        constexpr uint32_t mantissa_bits = 7;
        constexpr uint32_t exponent_size = (1U << exponent_bits);
        constexpr uint32_t mantissa_size = (1U << mantissa_bits);
        constexpr uint32_t table_size = (1U << (exponent_bits + mantissa_bits));

        rsqrt_param.init(table_size);
        const double out_scale = std::pow(2.0, 1.5 * static_cast<double>(scale_bit));

        for (uint32_t m = 0; m < mantissa_size; ++m)
        {
            const double mantissa = 1.0 + static_cast<double>(m) / static_cast<double>(mantissa_size);
            for (uint32_t e = 0; e < exponent_size; ++e)
            {
                const uint32_t index = (m << exponent_bits) | e;
                const double x_norm = std::ldexp(mantissa, static_cast<int>(e));
                const double y = out_scale / std::sqrt(x_norm);
                rsqrt_param.table.data[index] = static_cast<T>(std::llround(y));
            }
        }

        uint32_t sqrt_nexpb_table_size = (int)(log(pow(2, 2 * scale_bit)) / log(SCALE_BASE)) + 1;;
        sqrt_nexpb_param.init(sqrt_nexpb_table_size);
        for(int i = 0; i < sqrt_nexpb_table_size; i++)
        {
            sqrt_nexpb_param.table.data[i] = (T)(sqrt(pow(2, scale_bit) / pow(SCALE_BASE, i)) * (1 << scale_bit));
        }
    }

    void init_gelu()
    {
        // table_scale_bit = 6;
        uint32_t table_size = 4 * (1 << GELU_TABLE_PRECISION);
        float table_scale = 1 << GELU_TABLE_PRECISION;
        gelu_param.init(table_size);

        for (int i = 0; i < table_size; i++)
        {
            float real_value = i / table_scale;
            gelu_param.table.data[i] = (T)((MY_RELU(real_value) - GeLU(real_value)) * (1 << scale_bit));
        }
    }

    void init_swiglu()
    {
        constexpr uint32_t swiglu_table_range = 4;
        uint32_t table_size = swiglu_table_range * (1U << SWIGLU_TABLE_PRECISION);
        float table_scale = static_cast<float>(1U << SWIGLU_TABLE_PRECISION);
        swiglu_param.init(table_size);

        for (uint32_t i = 0; i < table_size; i++)
        {
            float real_value = static_cast<float>(i) / table_scale;
            swiglu_param.table.data[i] = static_cast<T>((MY_RELU(real_value) - SiLU(real_value)) * (1 << scale_bit));
        }
    }

    void init_all()
    {   
        fss_init();
        init_pc_cmp();
        init_nexp2();
        init_nexpb();
        init_exp();
        init_inv();
        init_inv_chopped(4096);
        init_rsqrt();
        init_gelu();
        init_swiglu();
    }
};

#endif // PARAMS_H