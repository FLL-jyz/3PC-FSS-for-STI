#ifndef PARAMS_H
#define PARAMS_H
#include "globals.h"
#include "my_math.h"
#include "replicated_secret_sharing.h"
#include "tensor.h"
#include "party3pc.h"
#include "dcf.h"
#include "dpf.h"
#include "fss.h"
#include <tuple>

// follow the pattern of debug level 2
template <typename T>
class DRelu_param
{
public:
    // DReLU parameters
    std::vector<DCFKeyPack> keys;
    std::vector<T> r_in_share; // RSS share of r
    std::vector<T> c_in_share; // RSS share of c
    std::vector<T> y_in_share; // RSS share of y

    void init(int party_id, bool real = false)
    {
        // DReLU init (Simulation)
        keys.resize(3);
        r_in_share.resize(3);
        c_in_share.resize(3);
        
        T r = 0;
        T c = 0;
        if (real) {
            r = (T)rand();
            c = rand() % 2;
            T r0 = (T)rand();
            T r1 = (T)rand();
            r_in_share = {r0, r1, (T)(r - r0 - r1)};
            T c0 = (T)rand();
            T c1 = (T)rand();
            c_in_share = {c0, c1, (T)(c - c0 - c1)};
        } else {
            //r_in,c_in should be random, but we set it to 0 for easy simulate.
            r_in_share = {0, 0, 0};
            c_in_share = {0, 0, 0};
        }
        y_in_share = {0, 0, 0};
        
        // y = 2^l - 1 - ((2^l - r) mod 2^{l-1})
        // With r=0, y = -1 (all 1s)
        T y;
        if (real) {
            T mask = ((T)1 << (sizeof(T) * 8 - 1)) - 1;
            T term = (-r) & mask;
            y = (T)(-1) - term;
        } else {
            y = (T)-1;
        }

        int bin = sizeof(T) * 8;
        int bout = 1;
        
        // Generate DCF keys for y
        GroupElement payload(1, bout);
        auto keys_gen = keyGenDCF(bin, bout, GroupElement(y, bin), payload);
        
        if (party_id == 0) keys[0] = std::get<0>(keys_gen);
        if (party_id == 1) keys[1] = std::get<1>(keys_gen);
        if (party_id == 2) keys[2] = std::get<2>(keys_gen);
    }
};

template <typename T>
class Trunc_param
{
public:
    std::vector<DCFKeyPack> keys;
    std::vector<T> r_share;
    std::vector<T> y_share;
    std::vector<T> y_shift_share;
    DRelu_param<T> *drelu;

    void init(int party_id, uint32_t s, DRelu_param<T> *drelu_ptr, bool real = false)
    {
        keys.resize(3);
        r_share.resize(3);
        y_share.resize(3);
        drelu = drelu_ptr;

        T r = 0; 
        if (real) {
            r = (T)rand();
            T r0 = (T)rand();
            T r1 = (T)rand();
            r_share = {r0, r1, (T)(r - r0 - r1)};
        } else {
            r_share = {0, 0, 0};
        }
        
        T y = (T)0 - r; 
        T alpha_s = y & ((1ULL << s) - 1); 
        
        if (real) {
            T y0 = (T)rand();
            T y1 = (T)rand();
            y_share = {y0, y1, (T)(y - y0 - y1)};
        } else {
            y_share = {y, 0, 0}; 
            if (party_id == 1) y_share[0] = 0; 
        }

        T y_shift = y >> s;
        if (real) {
            T y_shift0 = (T)rand();
            T y_shift1 = (T)rand();
            y_shift_share = {y_shift0, y_shift1, (T)(y_shift - y_shift0 - y_shift1)};
        } else {
            y_shift_share = {y_shift, 0, 0};
            if (party_id == 1) y_shift_share[0] = 0;
        }

        int bin = sizeof(T) * 8;
        GroupElement payload(1, 1);
        auto keys_trunc = keyGenDCF(bin, 1, GroupElement(alpha_s, bin), payload);
        if (party_id == 0) keys[0] = std::get<0>(keys_trunc);
        if (party_id == 1) keys[1] = std::get<1>(keys_trunc);
        if (party_id == 2) keys[2] = std::get<2>(keys_trunc);
    }
};

template <typename T>
class UCMP_Param {
public:
    std::vector<DCFKeyPack> keys;
    std::vector<T> r_in_share; // Share of r^in held by this party
    // Note: In a real offline phase, we would store pre-generated tuples.
    // Here we generate them on-the-fly based on y.

    void init(int party_id, const Tensor<T>& y, bool real = false) {
        uint32_t size = real ? y.size() : 1;
        keys.resize(size);
        r_in_share.resize(3);

        T r_in = 0;
        if (real) {
            r_in = (T)rand();
            T r0 = (T)rand();
            T r1 = (T)rand();
            r_in_share[0] = r0;
            r_in_share[1] = r1;
            r_in_share[2] = r_in - r0 - r1;
        } else {
            //r_in should be random, but we set it to 0 for easy simulate.
            r_in_share[0] = 0;
            r_in_share[1] = 0;
            r_in_share[2] = 0;
        }

        int bin = sizeof(T) * 8;
        int bout = 1; // Output is 1 bit (arithmetic share of 1)

        for(uint32_t i = 0; i < size; ++i) {
            // 4. Generate DCF keys (Keep existing logic if needed, though we use V now)
            T alpha = y.data[i] + r_in; 
            GroupElement payload(1, bout);
            auto keys_gen = keyGenDCF(bin, bout, GroupElement(alpha, bin), payload);

            if (party_id == 0) keys[i] = std::get<0>(keys_gen);
            if (party_id == 1) keys[i] = std::get<1>(keys_gen);
            if (party_id == 2) keys[i] = std::get<2>(keys_gen);
        }
    }
};

template <typename T>
class LUT_Param
{
public:
    Tensor<T> table;
    uint32_t table_size;

    std::vector<T> r_in_share;
    std::vector<DPFKeyPack> keys;
    std::vector<Tensor<T>> onehot_table;
    void init(uint32_t table_size)
    {
        this->table_size = table_size;
        table.allocate({table_size});
        r_in_share.resize(3, 0);
        onehot_table.resize(3);
        for(int i = 0; i < 3; ++i)
        {
            onehot_table[i].allocate({table_size});
            onehot_table[i].zero();
        }
        
        keys.resize(3);
    }
};

template <typename T>
class Parameters
{
public:
    DRelu_param<T> DRelu;
    Trunc_param<T> trunc_param;
    LUT_Param<T> nexp2_param;
    LUT_Param<T> nexpb_param;
    LUT_Param<T> inv_param;
    LUT_Param<T> sqrt_nexpb_param;
    LUT_Param<T> rsqrt_param;
    LUT_Param<T> gelu_param;
    UCMP_Param<T> ucmp_param;
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
        this->party_id = Party3PC::getInstance().party_id;
    }

    void init_DRelu(bool real = false)
    {
        DRelu.init(party_id, real);
    }

    void init_trunc(uint32_t s, bool real = false)
    {
        trunc_param.init(party_id, s, &this->DRelu, real);
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

    void init_rsqrt()
    {
        uint32_t table_size = (uint32_t)ceil((1.0  - 1.0 / SCALE_BASE) * (1 << scale_bit));  // 0.75 * (2 ^ f)
        float table_scale = 1 << scale_bit;
        rsqrt_param.init(table_size);

        for(int i = 0; i < table_size; i++)
        {
            float real_value = i / table_scale + 1.0 / SCALE_BASE;
            rsqrt_param.table.data[i] = (T)(sqrt(1 / real_value) * (1 << scale_bit));
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

    // Simulate Dealer for UCMP
    void init_ucmp(const Tensor<T>& y, bool real = false) {
        ucmp_param.init(party_id, y, real);
    }

    void init_all(bool real = false)
    {   
        fss_init();
        init_DRelu(real);
        init_nexp2();
        init_nexpb();
        init_inv();
        init_rsqrt();
        init_gelu();
    }

    void init_for_gelu(bool real = false)
    {   
        fss_init();
        init_DRelu(real);
        init_gelu();
    }
};

#endif // PARAMS_H