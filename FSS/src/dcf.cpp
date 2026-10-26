/*
Authors: Deepak Kumaraswamy, Kanav Gupta
Copyright:
Copyright (c) 2022 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "dcf.h"
#include <omp.h>
#include <iostream>
#include <cstdint>
#include <deque>

// ...existing code...
#ifndef FSS_FP_SCALE
#define FSS_FP_SCALE (1ULL<<10)
#endif

// 移除浮点数定义，直接使用整数计算
// using fp_t = int64_t;
// ...

using namespace osuCrypto;
// uint64_t aes_evals_count = 0;

#define SERVER0 0
#define SERVER1 1
#define SERVER2 2
#define GROUP_LOOP(s)                  \
    int lp = (evalGroupIdxStart + groupSize) % groupSize;        \
    int ctr = 0;                       \
    while(ctr < evalGroupIdxLen)       \
    {                                  \
        s                              \
        lp = (lp + 1) % groupSize;     \
        ctr++;                         \
    }

namespace {

constexpr int kFssParallelLoopThreshold = 1024;
constexpr uint64_t kFssParallelWorkThreshold = 1ULL << 12;

inline bool fss_should_parallelize_loop(uint64_t work_items, bool force_parallel = false)
{
    if (omp_get_max_threads() <= 1)
    {
        return false;
    }
    if (!force_parallel && omp_in_parallel())
    {
        return false;
    }
    return work_items >= kFssParallelLoopThreshold;
}

inline bool fss_should_parallelize_tree(int bin, int split_depth, int group_size, uint64_t range_len)
{
    if (omp_get_max_threads() <= 1 || split_depth <= 0)
    {
        return false;
    }

    const int effective_depth = std::min(bin, split_depth);
    if (effective_depth <= 1)
    {
        return false;
    }

    const int task_bits = std::min(effective_depth, 12);
    const uint64_t estimated_tasks = 1ULL << task_bits;
    const uint64_t estimated_work = range_len * static_cast<uint64_t>(std::max(1, group_size));
    return estimated_tasks >= static_cast<uint64_t>(std::max(4, omp_get_max_threads() / 2)) &&
           estimated_work >= kFssParallelWorkThreshold;
}

} // namespace

block traverseOneDCF(int Bin, int Bout, int groupSize, int party,
                        const block &s,
                        const block &cw1,
                        const block &cw2,
                        const u8 &keep,
                        GroupElement *v_share,
                        GroupElement *v,
                        uint64_t level,
                        bool geq,
                        int evalGroupIdxStart,
                        int evalGroupIdxLen,
                        bool forceParallel = false)

{
    static const block notThreeBlock = toBlock(~0, ~3);
    static const block notTwoBlock = toBlock(~0, ~2);
    static const block TwoBlock = toBlock(0, 2);
    static const block ThreeBlock = toBlock(0, 3);
    static const block blocks[4] = {ZeroBlock, TwoBlock, OneBlock, ThreeBlock};
    static const block BytesBlock = toBlock(0, 15);
    static const block notBytesBlock = toBlock(~0, ~15);

    block stcw;
    block ct[2]; // {tau, v_this_level}  存储当前s值和v值
    u8 t_previous = lsb(s);
    u8 u_previous = lsb(s>>1);
    auto scw = (cw1 & notBytesBlock);
    block t1_bits[] = { ((cw1 >> 1) & OneBlock), (cw1 & OneBlock) }; //取出t1cw
    block t2_bits[] = { ((cw2 >> 1) & OneBlock), (cw2 & OneBlock) }; //取出t2cw
    block u1_bits[] = { ((cw1 >> 3) & OneBlock), ((cw1 >> 2) & OneBlock) }; //取出u1cw
    block u2_bits[] = { ((cw2 >> 3) & OneBlock), ((cw2 >> 2) & OneBlock) }; //取出u2cw
    const auto mask1 = zeroAndAllOne[t_previous & u_previous];  //全0和全1掩码
    const auto mask2 = zeroAndAllOne[(t_previous ^ u_previous) & (t_previous | u_previous)];
    auto ss = s & notBytesBlock;
    if (t_previous & u_previous){
        scw = (cw2 & notBytesBlock);
    }

    AES ak(ss);
    ak.ecbEncTwoBlocks(blocks + 2 * keep, ct);
    if (t_previous ==0 && int(lsb(ct[0])) == 0){
        ct[0] = ct[0] & notTwoBlock;  // 将倒数第二位强制设置为0
    }
    stcw = ((scw ^ t2_bits[keep] ^ (u2_bits[keep] << 1)) & mask1)
            ^ ((scw ^ t1_bits[keep] ^ (u1_bits[keep] << 1)) & mask2)
            ^ ct[0];


    uint64_t sign = (party == SERVER1) ? -1 : 1;
    //判断0和2号服务器，和VCW的除2要加1
    uint64_t v_this_level_converted[groupSize];

    convert(Bout, groupSize, ct[1], v_this_level_converted);

    #pragma omp parallel for if(fss_should_parallelize_loop(evalGroupIdxLen, forceParallel))
    for(int ctr = 0; ctr < evalGroupIdxLen; ctr++)
    {
        int lp = (evalGroupIdxStart + ctr) % groupSize;
        v_this_level_converted[lp] &= 0x000000000FFFFFFFULL;
        GroupElement* vptr = (v + ((int)level) * groupSize + lp);
        uint64_t tmp = v_this_level_converted[lp];
        if (party == SERVER1){
            tmp = -2 * tmp;
        }
        if (t_previous == 1){
            if (party == SERVER1){
                tmp -= vptr->value;
            }
            else{
                uint64_t val = vptr->value;
                if(party == SERVER0 && (val & 1)){
                    val += 1;
                }
                if (val > (UINT64_MAX >> 1)){
                    val = (UINT64_MAX - ((UINT64_MAX - val)>>1));
                }
                else{
                    val >>= 1;
                }
                tmp += val;
            }
        }

        v_share[lp].value += tmp;
    }
    return stcw;
}


block traversePathDCF(int Bin, int Bout, int groupSize, int party,
                        GroupElement idx,
                        block *k0,
                        block *k1,
                        GroupElement *v_share,
                        GroupElement *v,
                        bool geq,
                        int evalGroupIdxStart,
                        int evalGroupIdxLen)
{
    block s = k0[0];
    #pragma omp parallel for if(!omp_in_parallel() && groupSize >= 1024)
    for(int lp = 0; lp < groupSize; ++lp) {
        if (lp >= evalGroupIdxStart && lp < evalGroupIdxStart + evalGroupIdxLen) {
             v_share[lp].value = 0;
        }
    }


    for (int i = 0; i < Bin; ++i)
    {
        const u8 keep = static_cast<uint8_t>(idx.value >> (Bin - 1 - i)) & 1;
        s = traverseOneDCF(Bin, Bout, groupSize, party, s, k0[i + 1], k1[i + 1], keep, v_share, v, i, geq, evalGroupIdxStart, evalGroupIdxLen);
    }
    return s;
}


// Real Endpoints
std::tuple<DCFKeyPack, DCFKeyPack, DCFKeyPack> keyGenDCF(int Bin, int Bout, int groupSize,
                GroupElement idx, GroupElement* payload, const GenParams &gen)
{
    // idx: bitsize Bin, payload: bitsize Bout & size groupSize
    bool greaterThan = false;
    static const block smallBlock = toBlock(0, ~0);
    static const block notOneBlock = toBlock(~0, ~1);
    static const block notThreeBlock = toBlock(~0, ~3);
    static const block notTwoBlock = toBlock(~0, ~2);
    static const block TwoBlock = toBlock(0, 2);
    static const block ThreeBlock = toBlock(0, 3);
    static const block BytesBlock = toBlock(0, 15);
    static const block notBytesBlock = toBlock(~0, ~15);
    const static block pt[4] = {ZeroBlock, OneBlock, TwoBlock, ThreeBlock};

    //三方DCF修改
    auto s = prng.get<std::array<block, 3>>();  //创建初始三个结点
    block si[3][2];
    block vi[3][2];
    //

    GroupElement *v_alpha = new GroupElement[groupSize];
    GroupElement *v_alpha1 = new GroupElement[groupSize];
#pragma omp parallel for if(!omp_in_parallel() && groupSize >= 1024)
    for (int i = 0; i < groupSize; ++i)
    {
        v_alpha[i] = GroupElement(0, Bout);
        v_alpha1[i] = GroupElement(0, Bout);
    }

    //三方DCF修改
    block *k0 = new block[Bin + 1];
    block *k1 = new block[Bin + 1];
    block *k2 = new block[Bin + 1];
    block *k_ = new block[Bin + 1];
    //
    GroupElement *v0 = new GroupElement[Bin * groupSize];    // bitsize Bout, size Bin x groupSize
    GroupElement *g0 = new GroupElement[groupSize];     // bitsize: Bout

    // 使用 uint64_t 替代 fp_t
    uint64_t *v_alpha_val = new uint64_t[groupSize];
    uint64_t *v0_val      = new uint64_t[Bin * groupSize];
#pragma omp parallel for if(!omp_in_parallel() && groupSize >= 1024)
    for(int i=0;i<groupSize;++i) v_alpha_val[i] = 0;
#pragma omp parallel for if(!omp_in_parallel() && (Bin * groupSize) >= 4096)
    for(int i=0;i<Bin * groupSize;++i){
        v0_val[i] = 0;
        v0[i] = GroupElement(0, Bout);
    }

    //三方DCF修改
    s[0] = (s[0] & notOneBlock) ^ ((s[1] & OneBlock) ^ OneBlock);  //使得t0^t1=为1
    s[2] = (s[2] & notOneBlock) ^ OneBlock;  //使得t2为1
    // 设置倒数第二位，使得 s[0]、s[1]、s[2] 的倒数第二位异或等于0，且只有倒数第一位为1时，倒数第二位才能为1
    for (int i = 0; i < 3; ++i) {
        uint8_t bit0 = lsb(s[i]);
        uint8_t bit1 = (i == 2) ? bit0 : 0;
        block low_bits = toBlock(0, (bit1 << 1) | bit0);
        s[i] = (s[i] & notThreeBlock) ^ low_bits;
    }
    k0[0] = s[0];
    k1[0] = s[1];
    k2[0] = s[2];
    block ct[4];
    //

    for (int i = 0; i < Bin; ++i)
    {
        const u8 keep = static_cast<uint8_t>(idx.value >> (Bin - 1 - i)) & 1;
        auto a = toBlock(keep);

        //三方DCF修改
        auto ss0 = s[0] & notBytesBlock;
        auto ss1 = s[1] & notBytesBlock;
        auto ss2 = s[2] & notBytesBlock;
        //

        AES ak0(ss0);
        AES ak1(ss1);
        AES ak2(ss2);
        ak0.ecbEncFourBlocks(pt, ct);
        si[0][0] = ct[0];
        si[0][1] = ct[1];
        vi[0][0] = ct[2];
        vi[0][1] = ct[3];
        ak1.ecbEncFourBlocks(pt, ct);
        si[1][0] = ct[0];
        si[1][1] = ct[1];
        vi[1][0] = ct[2];
        vi[1][1] = ct[3];
        ak2.ecbEncFourBlocks(pt, ct);
        si[2][0] = ct[0];
        si[2][1] = ct[1];
        vi[2][0] = ct[2];
        vi[2][1] = ct[3];

        auto ti0 = lsb(s[0]);
        auto ti1 = lsb(s[1]);
        auto ti2 = lsb(s[2]);
        //Take the second least significant bit
        auto ui0 = lsb(s[0]>>1);
        auto ui1 = lsb(s[1]>>1);
        auto ui2 = lsb(s[2]>>1);

        // 创建包含ti0、ui0、ti1、ui1、ti2和ui2的数组
        std::array<u8, 3> t_bits = {ti0, ti1, ti2};
        std::array<u8, 3> u_bits = {ui0, ui1, ui2};


        //这里改了t1的赋值，可能有问题
        // GroupElement sign(1, Bout);

        uint64_t vi_01_converted[groupSize];
        uint64_t vi_11_converted[groupSize];
        uint64_t vi_10_converted[groupSize];
        uint64_t vi_00_converted[groupSize];
        uint64_t vi_20_converted[groupSize];
        uint64_t vi_21_converted[groupSize];

#pragma omp parallel for if(!omp_in_parallel() && groupSize >= 1024)
        for(int i=0; i<groupSize; ++i) {
            vi_01_converted[i] = 0;
            vi_11_converted[i] = 0;
            vi_10_converted[i] = 0;
            vi_00_converted[i] = 0;
            vi_20_converted[i] = 0;
            vi_21_converted[i] = 0;
        }

        convert(Bout, groupSize, vi[0][keep], vi_00_converted);
        convert(Bout, groupSize, vi[1][keep], vi_10_converted);
        convert(Bout, groupSize, vi[0][keep ^ 1], vi_01_converted);
        convert(Bout, groupSize, vi[1][keep ^ 1], vi_11_converted);
        convert(Bout, groupSize, vi[2][keep], vi_20_converted);
        convert(Bout, groupSize, vi[2][keep ^ 1], vi_21_converted);
        // std::cout << "Max: " << UINT64_MAX<< std::endl;


        //创建两套纠正词
        int zeroIdx = -1;
        if (ti0 == 0) zeroIdx = 0;
        else if (ti1 == 0) zeroIdx = 1;
        else if (ti2 == 0) zeroIdx = 2;
        // 判断si[zeroIdx][keep]最后一位的值为0的话，强制要求倒数第二位的值也为0
        if (int(lsb(si[zeroIdx][keep])) == 0) {
            si[zeroIdx][keep] = si[zeroIdx][keep] & notTwoBlock;
        }
        // 找到ui和ti同时为1的情况
        int both_one_party = -1;
        if (ti0 == 1 && ui0 == 1) both_one_party = 0;
        else if (ti1 == 1 && ui1 == 1) both_one_party = 1;
        else if (ti2 == 1 && ui2 == 1) both_one_party = 2;
        // 找到ui和ti一个为1的情况
        int one_party = -1;
        if (ti0 ^ ui0 == 1) one_party = 0;
        else if (ti1 ^ ui1 == 1) one_party = 1;
        else if (ti2 ^ ui2 == 1) one_party = 2;
        if (int(lsb(si[zeroIdx][keep^1])) == 0) {
            si[zeroIdx][keep^1] = si[zeroIdx][keep^1] & notTwoBlock;
        }

        std::array<block, 2> siXOR1{si[zeroIdx][0] ^ si[one_party][0], si[zeroIdx][1] ^ si[one_party][1]};
        std::array<block, 2> siXOR2{si[zeroIdx][0] ^ si[both_one_party][0], si[zeroIdx][1] ^ si[both_one_party][1]};


        // get the left and right t_CW bits,并修改t值，避免α路径上出现两个t=0的情况
        std::array<block, 2> t1{ZeroBlock, ZeroBlock};
        std::array<block, 2> u1{ZeroBlock, ZeroBlock};
        std::array<block, 2> t2{ZeroBlock, ZeroBlock};
        std::array<block, 2> u2{ZeroBlock, ZeroBlock};
        std::array<int, 2> choose_list = {one_party, both_one_party};
        int randomValue = one_party;

        t1[0] = (OneBlock & siXOR1[0]) ^ a ^ OneBlock;
        t1[1] = (OneBlock & siXOR1[1]) ^ a;
        t2[0] = (OneBlock & siXOR2[0]) ^ a ^ OneBlock;
        t2[1] = (OneBlock & siXOR2[1]) ^ a;

        u1[0] = ((TwoBlock & siXOR1[0])>>1) ^ a ^ OneBlock;
        u1[1] = ((TwoBlock & siXOR1[1])>>1) ^ a;
        u2[0] = ((TwoBlock & siXOR2[0])>>1) ^ a ^ OneBlock;
        u2[1] = ((TwoBlock & siXOR2[1])>>1) ^ a;

        if (lsb(si[zeroIdx][keep])) {
                randomValue = choose_list[rand() % 2];  // 生成1或2的随机数
                if (randomValue == one_party) {
                    t1[keep] = (si[one_party][keep] & OneBlock) ^ OneBlock;
                }
                else {
                    t2[keep] = (si[both_one_party][keep] & OneBlock) ^ OneBlock;
                }
            }
        // check u, only when both t and u are 1, we need to correct u
        if(int(lsb(si[zeroIdx][keep])) == 1  && int(lsb(si[zeroIdx][keep]>>1)) == 0){
            if(randomValue == one_party) {
                u2[keep] = (si[both_one_party][keep] & TwoBlock) >> 1;
            }
            else {
                u1[keep] = (si[one_party][keep] & TwoBlock) >> 1;
            }
        }
        else if(int(lsb(si[zeroIdx][keep]>>1)) == 0  && int(lsb(si[zeroIdx][keep])) == 0){
            u2[keep] = (si[both_one_party][keep] & TwoBlock) >> 1;
        }

        // take scw to be the bits [127, 2] as scw = s0_loss ^ s1_loss
        auto scw1 = siXOR1[keep ^ 1] & notBytesBlock;
        auto scw2 = siXOR2[keep ^ 1] & notBytesBlock;
        k0[i + 1] = scw1 ^ (t1[0] << 1)
                        ^ (t1[1])
                        ^ (u1[0] << 3)
                        ^ (u1[1] << 2);
        k1[i + 1] = scw1 ^ (t1[0] << 1)
                        ^ (t1[1])
                        ^ (u1[0] << 3)
                        ^ (u1[1] << 2);
        k2[i + 1] = scw1 ^ (t1[0] << 1)
                        ^ (t1[1])
                        ^ (u1[0] << 3)
                        ^ (u1[1] << 2);
        k_[i + 1] = scw2 ^ (t2[0] << 1)
                        ^ (t2[1])
                        ^ (u2[0] << 3)
                        ^ (u2[1] << 2);


        auto si0Keep = si[zeroIdx][keep];
        auto si1Keep = si[one_party][keep];
        auto si2Keep = si[both_one_party][keep];

        // extract the t^Keep_CW bit
        auto TKeep1 = t1[keep];
        auto TKeep2 = t2[keep];
        auto UKeep1 = u1[keep] << 1;
        auto UKeep2 = u2[keep] << 1;

        s[zeroIdx] = si0Keep ^ (zeroAndAllOne[t_bits[zeroIdx] & u_bits[zeroIdx]] & (scw2 ^ TKeep2 ^ UKeep2))
                            ^ (zeroAndAllOne[t_bits[zeroIdx] ^ u_bits[zeroIdx]] & (scw1 ^ TKeep1 ^ UKeep1));

        s[one_party] = si1Keep ^ (zeroAndAllOne[t_bits[one_party] & u_bits[one_party]] & (scw2 ^ TKeep2 ^ UKeep2))
                            ^ (zeroAndAllOne[t_bits[one_party] ^ u_bits[one_party]] & (scw1 ^ TKeep1 ^ UKeep1));
        s[both_one_party] = si2Keep ^ (zeroAndAllOne[t_bits[both_one_party] & u_bits[both_one_party]] & (scw2 ^ TKeep2 ^ UKeep2))
                            ^ (zeroAndAllOne[t_bits[both_one_party] ^ u_bits[both_one_party]] & (scw1 ^ TKeep1 ^ UKeep1));
        //三方DCF修改,共用一套Vα，其中V0是VCW

        for (int lp = 0; lp < groupSize; ++lp)
        {
            vi_00_converted[lp] &= 0x000000000FFFFFFFULL;
            vi_01_converted[lp] &= 0x000000000FFFFFFFULL;
            vi_10_converted[lp] &= 0x000000000FFFFFFFULL;
            vi_11_converted[lp] &= 0x000000000FFFFFFFULL;
            vi_20_converted[lp] &= 0x000000000FFFFFFFULL;
            vi_21_converted[lp] &= 0x000000000FFFFFFFULL;

            if (ti1 == 1){
                // v0 = v0 + v2 - 2v1 + v_alpha
                v0_val[i * groupSize + lp] =
                      ((vi_01_converted[lp]
                    + vi_21_converted[lp]
                    - 2 * vi_11_converted[lp])<<1)
                    + v_alpha_val[lp];

                if (keep == 0 && greaterThan)
                    v0_val[i * groupSize + lp] -= (payload[lp].value * 2ULL);
                else if (keep == 1 && !greaterThan)
                    v0_val[i * groupSize + lp] -= (payload[lp].value * 2ULL);

                v_alpha_val[lp] = v_alpha_val[lp]
                               + ((- 2 * vi_10_converted[lp]
                               + vi_00_converted[lp]
                               + vi_20_converted[lp])<<1)
                               - v0_val[i * groupSize + lp];
                // v_alpha update
                if (int(lsb(s[1])) == 0){
                    if (v_alpha_val[lp] > (UINT64_MAX >> 1)){
                        v_alpha_val[lp] = (UINT64_MAX - ((UINT64_MAX - v_alpha_val[lp])>>1));
                    }
                    else{
                        v_alpha_val[lp] >>= 1;
                    }
                }

            } else {
                // ti1 == 0: v0 = 2v1 - v0 - v2 - v_alpha
                v0_val[i * groupSize + lp] =
                      2 * vi_11_converted[lp]
                    - vi_01_converted[lp]
                    - vi_21_converted[lp]
                    - v_alpha_val[lp];

                if (keep == 0 && greaterThan)
                    v0_val[i * groupSize + lp] += (payload[lp].value);
                else if (keep == 1 && !greaterThan)
                    v0_val[i * groupSize + lp] += (payload[lp].value);

                if (int(lsb(s[1])) == 1){
                    v_alpha_val[lp] = (v_alpha_val[lp]
                                     - 2 * vi_10_converted[lp]
                                     + vi_00_converted[lp]
                                     + vi_20_converted[lp]
                                     + v0_val[i * groupSize + lp]) * 2;
                } else {
                    v_alpha_val[lp] = v_alpha_val[lp]
                                   - 2 * vi_10_converted[lp]
                                   + vi_00_converted[lp]
                                   + vi_20_converted[lp]
                                   + v0_val[i * groupSize + lp];
                }
            }

            // 直接存值
            v0[i * groupSize + lp].value = v0_val[i * groupSize + lp];
            v_alpha[lp].value            = v_alpha_val[lp];
        }
    }
    uint64_t s0_converted[groupSize];
    uint64_t s1_converted[groupSize];
    uint64_t s2_converted[groupSize];
#pragma omp parallel for if(!omp_in_parallel() && groupSize >= 1024)
    for(int i=0; i<groupSize; ++i) {
        s0_converted[i] = 0;
        s1_converted[i] = 0;
        s2_converted[i] = 0;
    }
    convert(Bout, groupSize, s[0] & notBytesBlock, s0_converted);
    convert(Bout, groupSize, s[1] & notBytesBlock, s1_converted);
    convert(Bout, groupSize, s[2] & notBytesBlock, s2_converted);

    for (int lp = 0; lp < groupSize; ++lp)
    {
        s0_converted[lp] &= 0x000000000FFFFFFFULL;
        s1_converted[lp] &= 0x000000000FFFFFFFULL;
        s2_converted[lp] &= 0x000000000FFFFFFFULL;
        //  当t1=1时,cwn+1需要加上payload，并对s1的值修改 cw=2s1-s0-s2+2payload
        if (int(lsb(s[1])) == 1){
            // g0 = s0 + s2 - 2*s1 + v_alpha
            g0[lp].value = ((s0_converted[lp] + s2_converted[lp] - 2ULL*s1_converted[lp])<<1) + v_alpha_val[lp];
        }
        else {
            // g0 = 2*s1 - s0 - s2 - v_alpha
            g0[lp].value = 2ULL*s1_converted[lp] - s0_converted[lp] - s2_converted[lp] - v_alpha_val[lp];
        }
    }

    // Compute cs if requested
    std::array<std::array<block, 2>, 2> cs = {{{ZeroBlock, ZeroBlock}, {ZeroBlock, ZeroBlock}}};
    if (gen.gen_cs) {
        int zeroIdx = -1;
        if (lsb(s[0]) == 0) zeroIdx = 0;
        else if (lsb(s[1]) == 0) zeroIdx = 1;
        else if (lsb(s[2]) == 0) zeroIdx = 2;

        if (zeroIdx != -1) {
            auto pi_zero = Hash(idx.value, s[zeroIdx]);
            for(int i=0; i<3; ++i) {
                if (i == zeroIdx) continue;
                auto pi = Hash(idx.value, s[i]);
                int u = (int)lsb(s[i] >> 1);
                for(int k=0; k<2; ++k) {
                    cs[u][k] = pi[k] ^ pi_zero[k];
                }
            }
        }
    }

    return std::make_tuple(DCFKeyPack(Bin, Bout, groupSize, k0,k_,g0, v0, cs),DCFKeyPack(Bin, Bout, groupSize, k1,k_,g0, v0, cs), DCFKeyPack(Bin, Bout, groupSize, k2,k_,g0, v0, cs));
}

// std::tuple<DCFKeyPack, DCFKeyPack, DCFKeyPack> keyGenDCF(int Bin, int Bout,
//                 GroupElement idx, GroupElement payload)
// {
//     // idx: bitsize Bin, payload: bitsize Bout
//     return keyGenDCF(Bin, Bout, 1, idx, &payload);
// }

std::tuple<DCFKeyPack, DCFKeyPack,DCFKeyPack> keyGenDCF(int Bin, int Bout,
                GroupElement idx, GroupElement payload, const GenParams &gen)
{
    // 原来的两方DCF实现 - 需要重新实现或调用原来的逻辑
    auto result = keyGenDCF(Bin, Bout, 1, idx, &payload, gen);
    return std::make_tuple(std::get<0>(result), std::get<1>(result),std::get<2>(result));
}

void evalDCF(int Bin, int Bout, int groupSize,
                GroupElement *out, // groupSize
                int party, GroupElement idx,
                block *k0, // bin + 1
                block *k1,
                GroupElement *g , // groupSize
                GroupElement *v, // bin * groupSize
                bool geq /*= false*/, int evalGroupIdxStart /*= 0*/,
                int evalGroupIdxLen /*= -1*/,
                bool enableVerify /*= false*/,
                std::array<std::array<block, 2>, 2> cs /*= {}*/)
{
    if (evalGroupIdxLen == 0)
    {
        return;
    }
    if (evalGroupIdxLen == -1)
    {
        evalGroupIdxLen = groupSize;
    }

    auto s = traversePathDCF(Bin, Bout, groupSize, party, idx, k0, k1, out, v, geq, evalGroupIdxStart, evalGroupIdxLen);
    u8 t = lsb(s);

    if (enableVerify) {
        int u = (int)lsb(s >> 1);
        auto pi = Hash(idx.value, s);
        if (t) {
            for(int i=0; i<2; ++i) pi[i] = pi[i] ^ cs[u][i];
        }
        // Print verification tag for debugging/verification
        // std::cout << "Verification Tag (Party " << party << "): ";
        // for(int i=0; i<2; ++i) std::cout << pi[i] << " ";
        // std::cout << std::endl;
    }

    block temp = ZeroBlock;
    uint64_t sign0_2 = (party & t) ? -1 : 1;

    uint64_t s_converted[groupSize];
#pragma omp parallel for if(!omp_in_parallel() && groupSize >= 1024)
    for(int i=0; i<groupSize; ++i) s_converted[i] = 0;
    static const block notThreeBlock = toBlock(~0, ~3);
    static const block BytesBlock = toBlock(0, 15);
    static const block notBytesBlock = toBlock(~0, ~15);
    convert(Bout, groupSize, s & notBytesBlock, s_converted);

#pragma omp parallel for if(!omp_in_parallel() && evalGroupIdxLen >= 1024)
    for(int ctr = 0; ctr < evalGroupIdxLen; ++ctr)
    {
        int lp = (evalGroupIdxStart + ctr) % groupSize;
        s_converted[lp] &= 0x000000000FFFFFFFULL;
        uint64_t s_val = s_converted[lp];
        if (party == SERVER1){
            s_val = -2 * s_val;
        }
        if (t){
            uint64_t g_val = g[lp].value;
            if (party == SERVER1){
                s_val -= g_val;
            }
            else{
                if(party == SERVER0 && ((g_val & 1) == 1)){
                    g_val += 1;
                }
                if (g_val > (UINT64_MAX >> 1)){
                    g_val = (UINT64_MAX - ((UINT64_MAX - g_val)>>1));
                }
                else{
                    g_val >>= 1;
                }
                s_val += g_val;
            }
        }
        out[lp].value += s_val;
    }


}

void evalDCF(int party, GroupElement *res, GroupElement idx, const DCFKeyPack &key, bool enableVerify)
{
    evalDCF(key.Bin, key.Bout, key.groupSize, res, party, idx, key.k0, key.k1,key.g, key.v, false, 0, -1, enableVerify, key.cs);
}

void evalDCFPartial(int party, GroupElement *res, GroupElement idx, const DCFKeyPack &key, int start, int len)
{
    evalDCF(key.Bin, key.Bout, key.groupSize, res, party, idx, key.k0, key.k1,key.g, key.v, false, start, len);
}

// Dual DCF

std::pair<DualDCFKeyPack, DualDCFKeyPack> keyGenDualDCF(int Bin, int Bout, int groupSize, GroupElement idx, GroupElement *payload1, GroupElement *payload2)
{
    DualDCFKeyPack key0, key1;

    key0.Bin = Bin; key1.Bin = Bin;
    key0.Bout = Bout; key1.Bout = Bout;
    key0.groupSize = groupSize; key1.groupSize = groupSize;

    GroupElement *payload = new GroupElement[groupSize];
#pragma omp parallel for if(!omp_in_parallel() && groupSize >= 1024)
    for (int i = 0; i < groupSize; i++) {
        payload[i] = payload1[i] - payload2[i];
    }

    auto keys = keyGenDCF(Bin, Bout, groupSize, idx, payload);
    key0.dcfKey = std::get<0>(keys),  key1.dcfKey = std::get<1>(keys);


    key0.sb = new GroupElement[groupSize];
    key1.sb = new GroupElement[groupSize];

#pragma omp parallel for if(!omp_in_parallel() && groupSize >= 1024)
    for (int i = 0; i < groupSize; i++) {
        auto payload2_split = splitShare(payload2[i]);
        key0.sb[i] = payload2_split.first; key1.sb[i] = payload2_split.second;
    }

    return std::make_pair(key0, key1);
}

std::pair<DualDCFKeyPack, DualDCFKeyPack> keyGenDualDCF(int Bin, int Bout, GroupElement idx, GroupElement payload1, GroupElement payload2)
{
    return keyGenDualDCF(Bin, Bout, 1, idx, &payload1, &payload2);
}

void evalDualDCF(int party, GroupElement* res, GroupElement idx, const DualDCFKeyPack &key)
{
    evalDCF(key.Bin, key.Bout, key.groupSize, res, party, idx, key.dcfKey.k0,key.dcfKey.k1, key.dcfKey.g, key.dcfKey.v);
    for (int i = 0; i < key.groupSize; i++) {
        res[i] = res[i] + key.sb[i];
    }
}

static inline uint64_t fss_pow2_u64(int bits)
{
    return (bits >= 64 || bits < 0) ? 0ULL : (1ULL << bits);
}

static inline int fss_floor_log2_u64(uint64_t x)
{
    if (x == 0) return -1;
    return 63 - __builtin_clzll(x);
}

static inline int fss_recommended_split_depth(int Bin, uint64_t rangeLen)
{
    int d = std::max(0, Bin - 20);
    d = std::max(d, 10);
    d = std::min(d, 16);
    d = std::min(d, Bin);

    if (rangeLen > 0) {
        int r = fss_floor_log2_u64(rangeLen);
        if (r >= 0) {
            int minD = std::max(0, Bin - r);
            d = std::max(d, std::min(minD, 16));
            d = std::min(d, Bin);
        }
    }
    return d;
}

static inline bool fss_range_disjoint(uint64_t aStart, uint64_t aLen, uint64_t bStart, uint64_t bLen)
{
    uint64_t aEnd = aStart + aLen;
    uint64_t bEnd = bStart + bLen;
    return (aEnd <= bStart) || (bEnd <= aStart);
}

static void traverseRangeDCFSerial(int Bin, int Bout, int groupSize, int party,
                                   block s,
                                   block *k0, block *k1,
                                   GroupElement *v,
                                   GroupElement *g,
                                   GroupElement *v_share_stack,
                                   GroupElement *res,
                                   uint64_t rangeStart, uint64_t rangeLen,
                                   uint64_t pathIdx, int level)
{
    if (rangeLen == 0) return;
    const uint64_t subtreeLen = fss_pow2_u64(Bin - level);
    const uint64_t subtreeStart = pathIdx << (Bin - level);
    if (subtreeLen == 0 || fss_range_disjoint(subtreeStart, subtreeLen, rangeStart, rangeLen)) {
        return;
    }

    GroupElement *curr_v_share = &v_share_stack[level * groupSize];

    if (level == Bin) {
        const uint64_t leafIdx = pathIdx;
        const uint64_t off = leafIdx - rangeStart;
        u8 t = lsb(s);
        uint64_t s_converted[groupSize];
        static const block notBytesBlock = toBlock(~0, ~15);
        convert(Bout, groupSize, s & notBytesBlock, s_converted);

#pragma omp parallel for if(!omp_in_parallel() && groupSize >= 1024)
        for(int lp=0; lp<groupSize; ++lp){
            GroupElement final_val = curr_v_share[lp];
            s_converted[lp] &= 0x000000000FFFFFFFULL;
            uint64_t s_val = s_converted[lp];
            if (party == SERVER1){
                s_val = -2 * s_val;
            }
            if (t){
                uint64_t g_val = g[lp].value;
                if (party == SERVER1){
                    s_val -= g_val;
                }
                else{
                    if(party == SERVER0 && ((g_val & 1) == 1)){
                        g_val += 1;
                    }
                    if (g_val > (UINT64_MAX >> 1)){
                        g_val = (UINT64_MAX - ((UINT64_MAX - g_val)>>1));
                    }
                    else{
                        g_val >>= 1;
                    }
                    s_val += g_val;
                }
            }
            final_val.value += s_val;
            res[off * groupSize + lp].value += final_val.value;
        }
        return;
    }

    GroupElement *next_v_share = &v_share_stack[(level + 1) * groupSize];
    const uint64_t childLen = subtreeLen >> 1;
    const uint64_t leftStart = subtreeStart;
    const uint64_t rightStart = subtreeStart + childLen;

    if (!fss_range_disjoint(leftStart, childLen, rangeStart, rangeLen)) {
        for(int i=0; i<groupSize; ++i) next_v_share[i] = curr_v_share[i];
        block s_left = s;
        s_left = traverseOneDCF(Bin, Bout, groupSize, party, s_left, k0[level + 1], k1[level + 1], 0,
                                next_v_share, v, level, false, 0, groupSize);
        traverseRangeDCFSerial(Bin, Bout, groupSize, party, s_left, k0, k1, v, g, v_share_stack, res,
                               rangeStart, rangeLen, pathIdx << 1, level + 1);
    }

    if (!fss_range_disjoint(rightStart, childLen, rangeStart, rangeLen)) {
        for(int i=0; i<groupSize; ++i) next_v_share[i] = curr_v_share[i];
        block s_right = s;
        s_right = traverseOneDCF(Bin, Bout, groupSize, party, s_right, k0[level + 1], k1[level + 1], 1,
                                 next_v_share, v, level, false, 0, groupSize);
        traverseRangeDCFSerial(Bin, Bout, groupSize, party, s_right, k0, k1, v, g, v_share_stack, res,
                               rangeStart, rangeLen, (pathIdx << 1) | 1, level + 1);
    }
}

struct DCFNode {
    block s;
    uint64_t path;
    int level;
    const GroupElement* v_share;
};

static void collectDCFNodesSerial(int Bin, int Bout, int groupSize, int party,
                                  block s, block *k0, block *k1,
                                  GroupElement *v,
                                  GroupElement *v_share_stack,
                                  uint64_t rangeStart, uint64_t rangeLen,
                                  uint64_t pathIdx, int level, int splitDepth,
                                  std::vector<std::vector<DCFNode>>& thread_nodes,
                                  std::vector<std::deque<GroupElement>>& thread_data);

static void collectDCFNodes(int Bin, int Bout, int groupSize, int party,
                            block s, block *k0, block *k1,
                            GroupElement *v,
                            const std::vector<GroupElement>& current_v_share,
                            uint64_t rangeStart, uint64_t rangeLen,
                            uint64_t pathIdx, int level, int splitDepth,
                            std::vector<std::vector<DCFNode>>& thread_nodes,
                            std::vector<std::deque<GroupElement>>& thread_data)
{
    // Pruning
    const uint64_t subtreeLen = fss_pow2_u64(Bin - level);
    const uint64_t subtreeStart = pathIdx << (Bin - level);
    if (subtreeLen == 0 || fss_range_disjoint(subtreeStart, subtreeLen, rangeStart, rangeLen)) {
        return;
    }

    // Switch to Serial if depth is deep enough (avoid tasks and allocations)
    if (level >= splitDepth - 4) {
        std::vector<GroupElement> stack((splitDepth + 2) * groupSize);
        // Copy initial state to stack
        for(int i=0; i<groupSize; ++i) stack[level * groupSize + i] = current_v_share[i];

        collectDCFNodesSerial(Bin, Bout, groupSize, party, s, k0, k1, v, stack.data(),
                              rangeStart, rangeLen, pathIdx, level, splitDepth, thread_nodes, thread_data);
        return;
    }

    if (level == splitDepth || level == Bin) {
        int tid = omp_get_thread_num();
        // Use padding (x8)
        auto& deque = thread_data[tid * 8];
        size_t start_idx = deque.size();
        for(const auto& elem : current_v_share) deque.push_back(elem);

        thread_nodes[tid * 8].push_back({s, pathIdx, level, &deque[start_idx]});
        return;
    }

    // Recurse Left
    #pragma omp task shared(thread_nodes, thread_data) if(level < splitDepth - 4)
    {
        std::vector<GroupElement> next_v_share_left = current_v_share;
        block s_left = s;
        // forceParallel = true because we want to use threads for large groupSize even inside omp task
        s_left = traverseOneDCF(Bin, Bout, groupSize, party, s_left, k0[level + 1], k1[level + 1], 0,
                                next_v_share_left.data(), v, level, false, 0, groupSize, true);
        collectDCFNodes(Bin, Bout, groupSize, party, s_left, k0, k1, v, next_v_share_left,
                        rangeStart, rangeLen, pathIdx << 1, level + 1, splitDepth, thread_nodes, thread_data);
    }

    // Recurse Right
    #pragma omp task shared(thread_nodes, thread_data) if(level < splitDepth - 4)
    {
        std::vector<GroupElement> next_v_share_right = current_v_share;
        block s_right = s;
        s_right = traverseOneDCF(Bin, Bout, groupSize, party, s_right, k0[level + 1], k1[level + 1], 1,
                                 next_v_share_right.data(), v, level, false, 0, groupSize, true);
        collectDCFNodes(Bin, Bout, groupSize, party, s_right, k0, k1, v, next_v_share_right,
                        rangeStart, rangeLen, (pathIdx << 1) | 1, level + 1, splitDepth, thread_nodes, thread_data);
    }
    #pragma omp taskwait
}

static void collectDCFNodesSerial(int Bin, int Bout, int groupSize, int party,
                                  block s, block *k0, block *k1,
                                  GroupElement *v,
                                  GroupElement *v_share_stack,
                                  uint64_t rangeStart, uint64_t rangeLen,
                                  uint64_t pathIdx, int level, int splitDepth,
                                  std::vector<std::vector<DCFNode>>& thread_nodes,
                                  std::vector<std::deque<GroupElement>>& thread_data)
{
    // Pruning
    const uint64_t subtreeLen = fss_pow2_u64(Bin - level);
    const uint64_t subtreeStart = pathIdx << (Bin - level);
    if (subtreeLen == 0 || fss_range_disjoint(subtreeStart, subtreeLen, rangeStart, rangeLen)) {
        return;
    }

    if (level == splitDepth || level == Bin) {
        int tid = omp_get_thread_num();

        auto& deque = thread_data[tid * 8];
        size_t start_idx = deque.size();
        GroupElement* curr_v = &v_share_stack[level * groupSize];
        for(int i=0; i<groupSize; ++i) deque.push_back(curr_v[i]);

        // Safe to take address of element in deque (stable references)
        thread_nodes[tid * 8].push_back({s, pathIdx, level, &deque[start_idx]});
        return;
    }

    GroupElement *curr_v_share = &v_share_stack[level * groupSize];
    GroupElement *next_v_share = &v_share_stack[(level + 1) * groupSize];

    // Left
    {
        for(int i=0; i<groupSize; ++i) next_v_share[i] = curr_v_share[i];
        block s_left = s;
        // forceParallel = false (serial execution)
        s_left = traverseOneDCF(Bin, Bout, groupSize, party, s_left, k0[level + 1], k1[level + 1], 0,
                                next_v_share, v, level, false, 0, groupSize, false);
        collectDCFNodesSerial(Bin, Bout, groupSize, party, s_left, k0, k1, v, v_share_stack,
                              rangeStart, rangeLen, pathIdx << 1, level + 1, splitDepth, thread_nodes, thread_data);
    }

    // Right
    {
        for(int i=0; i<groupSize; ++i) next_v_share[i] = curr_v_share[i];
        block s_right = s;
        // forceParallel = false
        s_right = traverseOneDCF(Bin, Bout, groupSize, party, s_right, k0[level + 1], k1[level + 1], 1,
                                 next_v_share, v, level, false, 0, groupSize, false);
        collectDCFNodesSerial(Bin, Bout, groupSize, party, s_right, k0, k1, v, v_share_stack,
                              rangeStart, rangeLen, (pathIdx << 1) | 1, level + 1, splitDepth, thread_nodes, thread_data);
    }
}


static void traverseRangeDCFSerial_GS1(int Bin, int Bout, int party,
                                   block s,
                                   block *k0, block *k1,
                                   GroupElement *v,
                                   GroupElement *g,
                                   GroupElement *v_share_stack,
                                   GroupElement *res,
                                   uint64_t rangeStart, uint64_t rangeLen,
                                   uint64_t pathIdx, int level)
{
    if (rangeLen == 0) return;
    const uint64_t subtreeLen = fss_pow2_u64(Bin - level);
    const uint64_t subtreeStart = pathIdx << (Bin - level);
    if (subtreeLen == 0 || fss_range_disjoint(subtreeStart, subtreeLen, rangeStart, rangeLen)) {
        return;
    }

    // Inlined traverseOneDCF logic for GroupSize=1
    // Common constants
    static const block notBytesBlock = toBlock(~0, ~15);
    static const block notTwoBlock = toBlock(~0, ~2);
    static const block OneBlock = toBlock(0, 1);

    // Process current node correction (v_share update)
    // Note: traverseOneDCF updates v_share for the *next* level based on *current* s.
    // Wait, traverseOneDCF computes v_share for the children?
    // "ct[1]" is part of calculation.
    // Actually, traverseOneDCF is called to compute s_left/s_right and update their v_share.
    // But here we are at 'level'. We need to process children to recurse.

    if (level == Bin) {
        // Leaf logic
        const uint64_t leafIdx = pathIdx;
        const uint64_t off = leafIdx - rangeStart;
        u8 t = lsb(s);
        uint64_t s_val = 0;

        // Convert (casting for 64bit)
        block s_masked = s & notBytesBlock;
        s_val = _mm_cvtsi128_si64x(s_masked); // Equivalent to casting for first 64 bits
        s_val &= 0x000000000FFFFFFFULL;

        GroupElement final_val = v_share_stack[level]; // GroupSize=1, index=level

        if (party == SERVER1){
            s_val = -2 * s_val;
        }
        if (t){
            uint64_t g_val = g[0].value;
            if (party == SERVER1){
                s_val -= g_val;
            }
            else{
                if(party == SERVER0 && ((g_val & 1) == 1)){
                    g_val += 1;
                }
                if (g_val > (UINT64_MAX >> 1)){
                    g_val = (UINT64_MAX - ((UINT64_MAX - g_val)>>1));
                }
                else{
                    g_val >>= 1;
                }
                s_val += g_val;
            }
        }
        final_val.value += s_val;
        res[off].value += final_val.value;
        return;
    }

    // Recursive Step: Compute Left and Right Children
    // Replaces traverseOneDCF calls

    // 1. Prepare common data
    u8 t_previous = lsb(s);
    u8 u_previous = lsb(s>>1);
    block scw = k0[level + 1]; // k0 is cw array? No, k0 in traverseOne param is cw1.
    // In traverseRange call: k0[level+1], k1[level+1].
    // Note: Argument names in this function: k0, k1 are arrays of blocks.
    block cw1 = k0[level + 1];
    block cw2 = k1[level + 1];

    // static const block notBytesBlock = toBlock(~0, ~15);
    // static const block OneBlock = toBlock(0, 1);
    static const block zeroAndAllOne[2] = {ZeroBlock, toBlock(~0, ~0)};
    // static const block notTwoBlock = toBlock(~0, ~2);

    {
        // LEFT (keep=0)
        block s_left;
        u8 keep = 0;

        block scw_local = (cw1 & notBytesBlock);
        if (t_previous & u_previous){
            scw_local = (cw2 & notBytesBlock);
        }

        block t1_bits_keep = ((cw1 >> 1) & OneBlock);
        block t2_bits_keep = ((cw2 >> 1) & OneBlock);
        block u1_bits_keep = ((cw1 >> 3) & OneBlock);
        block u2_bits_keep = ((cw2 >> 3) & OneBlock);

        const auto mask1 = zeroAndAllOne[t_previous & u_previous];
        const auto mask2 = zeroAndAllOne[(t_previous ^ u_previous) & (t_previous | u_previous)];
        auto ss = s & notBytesBlock;

        block ct[2];
        AES ak(ss);
        static const block TwoBlock = toBlock(0, 2);
        block pt[2] = {ZeroBlock, TwoBlock};
        ak.ecbEncTwoBlocks(pt, ct);

        if (t_previous == 0 && int(lsb(ct[0])) == 0){
            ct[0] = ct[0] & notTwoBlock;
        }

        s_left = ((scw_local ^ t2_bits_keep ^ (u2_bits_keep << 1)) & mask1)
                ^ ((scw_local ^ t1_bits_keep ^ (u1_bits_keep << 1)) & mask2)
                ^ ct[0];

        uint64_t v_converted = _mm_cvtsi128_si64x(ct[1]);
        v_converted &= 0x000000000FFFFFFFULL;

        uint64_t tmp = v_converted;
        GroupElement* vptr = (v + ((int)level) * 1);

        if (party == SERVER1){
            tmp = -2 * tmp;
        }
        if (t_previous == 1){
            if (party == SERVER1){
                tmp -= vptr->value;
            }
            else{
                uint64_t val = vptr->value;
                if(party == SERVER0 && (val & 1)){
                    val += 1;
                }
                if (val > (UINT64_MAX >> 1)){
                    val = (UINT64_MAX - ((UINT64_MAX - val)>>1));
                }
                else{
                    val >>= 1;
                }
                tmp += val;
            }
        }

        v_share_stack[level + 1].value = v_share_stack[level].value + tmp;
        traverseRangeDCFSerial_GS1(Bin, Bout, party, s_left, k0, k1, v, g, v_share_stack, res, rangeStart, rangeLen, pathIdx << 1, level + 1);
    }

    {
        // RIGHT (keep=1)
        block s_right;
        u8 keep = 1;

        block scw_local = (cw1 & notBytesBlock);
        if (t_previous & u_previous){
            scw_local = (cw2 & notBytesBlock);
        }

        block t1_bits_keep = (cw1 & OneBlock);
        block t2_bits_keep = (cw2 & OneBlock);
        block u1_bits_keep = ((cw1 >> 2) & OneBlock);
        block u2_bits_keep = ((cw2 >> 2) & OneBlock);

        const auto mask1 = zeroAndAllOne[t_previous & u_previous];
        const auto mask2 = zeroAndAllOne[(t_previous ^ u_previous) & (t_previous | u_previous)];
        auto ss = s & notBytesBlock;

        block ct[2];
        AES ak(ss);
        static const block OneBlock = toBlock(0, 1);
        static const block ThreeBlock = toBlock(0, 3);
        block pt[2] = {OneBlock, ThreeBlock};
        ak.ecbEncTwoBlocks(pt, ct);

        if (t_previous == 0 && int(lsb(ct[0])) == 0){
            ct[0] = ct[0] & notTwoBlock;
        }

        s_right = ((scw_local ^ t2_bits_keep ^ (u2_bits_keep << 1)) & mask1)
                ^ ((scw_local ^ t1_bits_keep ^ (u1_bits_keep << 1)) & mask2)
                ^ ct[0];

        uint64_t v_converted = _mm_cvtsi128_si64x(ct[1]);
        v_converted &= 0x000000000FFFFFFFULL;

        uint64_t tmp = v_converted;
        GroupElement* vptr = (v + ((int)level) * 1);

        if (party == SERVER1){
            tmp = -2 * tmp;
        }
        if (t_previous == 1){
            if (party == SERVER1){
                tmp -= vptr->value;
            }
            else{
                uint64_t val = vptr->value;
                if(party == SERVER0 && (val & 1)){
                    val += 1;
                }
                if (val > (UINT64_MAX >> 1)){
                    val = (UINT64_MAX - ((UINT64_MAX - val)>>1));
                }
                else{
                    val >>= 1;
                }
                tmp += val;
            }
        }

        v_share_stack[level + 1].value = v_share_stack[level].value + tmp;
        traverseRangeDCFSerial_GS1(Bin, Bout, party, s_right, k0, k1, v, g, v_share_stack, res, rangeStart, rangeLen, (pathIdx << 1) | 1, level + 1);
    }
}

static void traverseRangeSumDCFSerial(int Bin, int Bout, int groupSize, int party,
                                   block s,
                                   block *k0, block *k1,
                                   GroupElement *v,
                                   GroupElement *g,
                                   GroupElement *v_share_stack,
                                   GroupElement *sums,
                                   uint64_t rangeStart, uint64_t rangeLen,
                                   uint64_t pathIdx, int level)
{
    if (rangeLen == 0) return;
    const uint64_t subtreeLen = fss_pow2_u64(Bin - level);
    const uint64_t subtreeStart = pathIdx << (Bin - level);
    if (subtreeLen == 0 || fss_range_disjoint(subtreeStart, subtreeLen, rangeStart, rangeLen)) {
        return;
    }

    GroupElement *curr_v_share = &v_share_stack[level * groupSize];

    if (level == Bin) {
        const uint64_t leafIdx = pathIdx;
        u8 t = lsb(s);
        uint64_t s_converted[groupSize];
        static const block notBytesBlock = toBlock(~0, ~15);
        convert(Bout, groupSize, s & notBytesBlock, s_converted);

        for(int lp=0; lp<groupSize; ++lp){
            GroupElement final_val = curr_v_share[lp];
            s_converted[lp] &= 0x000000000FFFFFFFULL;
            uint64_t s_val = s_converted[lp];
            if (party == SERVER1){
                s_val = -2 * s_val;
            }
            if (t){
                uint64_t g_val = g[lp].value;
                if (party == SERVER1){
                    s_val -= g_val;
                }
                else{
                    if(party == SERVER0 && ((g_val & 1) == 1)){
                        g_val += 1;
                    }
                    if (g_val > (UINT64_MAX >> 1)){
                        g_val = (UINT64_MAX - ((UINT64_MAX - g_val)>>1));
                    }
                    else{
                        g_val >>= 1;
                    }
                    s_val += g_val;
                }
            }
            final_val.value += s_val;
            sums[lp].value += final_val.value;
        }
        return;
    }

    GroupElement *next_v_share = &v_share_stack[(level + 1) * groupSize];
    const uint64_t childLen = subtreeLen >> 1;
    const uint64_t leftStart = subtreeStart;
    const uint64_t rightStart = subtreeStart + childLen;

    if (!fss_range_disjoint(leftStart, childLen, rangeStart, rangeLen)) {
        for(int i=0; i<groupSize; ++i) next_v_share[i] = curr_v_share[i];
        block s_left = s;
        s_left = traverseOneDCF(Bin, Bout, groupSize, party, s_left, k0[level + 1], k1[level + 1], 0,
                                next_v_share, v, level, false, 0, groupSize);
        traverseRangeSumDCFSerial(Bin, Bout, groupSize, party, s_left, k0, k1, v, g, v_share_stack, sums,
                               rangeStart, rangeLen, pathIdx << 1, level + 1);
    }

    if (!fss_range_disjoint(rightStart, childLen, rangeStart, rangeLen)) {
        for(int i=0; i<groupSize; ++i) next_v_share[i] = curr_v_share[i];
        block s_right = s;
        s_right = traverseOneDCF(Bin, Bout, groupSize, party, s_right, k0[level + 1], k1[level + 1], 1,
                                 next_v_share, v, level, false, 0, groupSize);
        traverseRangeSumDCFSerial(Bin, Bout, groupSize, party, s_right, k0, k1, v, g, v_share_stack, sums,
                               rangeStart, rangeLen, (pathIdx << 1) | 1, level + 1);
    }
}

static void traverseRangeSumDCFSerial_GS1(int Bin, int Bout, int party,
                                   block s,
                                   block *k0, block *k1,
                                   GroupElement *v,
                                   GroupElement *g,
                                   GroupElement *v_share_stack,
                                   GroupElement *sums,
                                   uint64_t rangeStart, uint64_t rangeLen,
                                   uint64_t pathIdx, int level)
{
    if (rangeLen == 0) return;
    const uint64_t subtreeLen = fss_pow2_u64(Bin - level);
    const uint64_t subtreeStart = pathIdx << (Bin - level);
    if (subtreeLen == 0 || fss_range_disjoint(subtreeStart, subtreeLen, rangeStart, rangeLen)) {
        return;
    }

    if (level == Bin) {
        const uint64_t leafIdx = pathIdx;
        u8 t = lsb(s);
        uint64_t s_val = 0;

        static const block notBytesBlock = toBlock(~0, ~15);
        block s_masked = s & notBytesBlock;
        s_val = _mm_cvtsi128_si64x(s_masked);
        s_val &= 0x000000000FFFFFFFULL;

        GroupElement final_val = v_share_stack[level];

        if (party == SERVER1){
            s_val = -2 * s_val;
        }
        if (t){
            uint64_t g_val = g[0].value;
            if (party == SERVER1){
                s_val -= g_val;
            }
            else{
                if(party == SERVER0 && ((g_val & 1) == 1)){
                    g_val += 1;
                }
                if (g_val > (UINT64_MAX >> 1)){
                    g_val = (UINT64_MAX - ((UINT64_MAX - g_val)>>1));
                }
                else{
                    g_val >>= 1;
                }
                s_val += g_val;
            }
        }
        final_val.value += s_val;
        sums[0].value += final_val.value;
        return;
    }

    // Inlined recursion logic for GS1
    u8 t_previous = lsb(s);
    u8 u_previous = lsb(s>>1);
    block cw1 = k0[level + 1];
    block cw2 = k1[level + 1];

    static const block notBytesBlock = toBlock(~0, ~15);
    static const block OneBlock = toBlock(0, 1);
    static const block zeroAndAllOne[2] = {ZeroBlock, toBlock(~0, ~0)};
    static const block notTwoBlock = toBlock(~0, ~2);

    {
        // LEFT (keep=0)
        block s_left;
        u8 keep = 0;

        block scw_local = (cw1 & notBytesBlock);
        if (t_previous & u_previous){
            scw_local = (cw2 & notBytesBlock);
        }

        block t1_bits_keep = ((cw1 >> 1) & OneBlock);
        block t2_bits_keep = ((cw2 >> 1) & OneBlock);
        block u1_bits_keep = ((cw1 >> 3) & OneBlock);
        block u2_bits_keep = ((cw2 >> 3) & OneBlock);

        const auto mask1 = zeroAndAllOne[t_previous & u_previous];
        const auto mask2 = zeroAndAllOne[(t_previous ^ u_previous) & (t_previous | u_previous)];
        auto ss = s & notBytesBlock;

        block ct[2];
        AES ak(ss);
        static const block TwoBlock = toBlock(0, 2);
        block pt[2] = {ZeroBlock, TwoBlock};
        ak.ecbEncTwoBlocks(pt, ct);

        if (t_previous == 0 && int(lsb(ct[0])) == 0){
            ct[0] = ct[0] & notTwoBlock;
        }

        s_left = ((scw_local ^ t2_bits_keep ^ (u2_bits_keep << 1)) & mask1)
                ^ ((scw_local ^ t1_bits_keep ^ (u1_bits_keep << 1)) & mask2)
                ^ ct[0];

        uint64_t v_converted = _mm_cvtsi128_si64x(ct[1]);
        v_converted &= 0x000000000FFFFFFFULL;

        uint64_t tmp = v_converted;
        GroupElement* vptr = (v + ((int)level) * 1);

        if (party == SERVER1){
            tmp = -2 * tmp;
        }
        if (t_previous == 1){
            if (party == SERVER1){
                tmp -= vptr->value;
            }
            else{
                uint64_t val = vptr->value;
                if(party == SERVER0 && (val & 1)){
                    val += 1;
                }
                if (val > (UINT64_MAX >> 1)){
                    val = (UINT64_MAX - ((UINT64_MAX - val)>>1));
                }
                else{
                    val >>= 1;
                }
                tmp += val;
            }
        }

        v_share_stack[level + 1].value = v_share_stack[level].value + tmp;
        traverseRangeSumDCFSerial_GS1(Bin, Bout, party, s_left, k0, k1, v, g, v_share_stack, sums, rangeStart, rangeLen, pathIdx << 1, level + 1);
    }

    {
        // RIGHT (keep=1)
        block s_right;
        u8 keep = 1;

        block scw_local = (cw1 & notBytesBlock);
        if (t_previous & u_previous){
            scw_local = (cw2 & notBytesBlock);
        }

        block t1_bits_keep = (cw1 & OneBlock);
        block t2_bits_keep = (cw2 & OneBlock);
        block u1_bits_keep = ((cw1 >> 2) & OneBlock);
        block u2_bits_keep = ((cw2 >> 2) & OneBlock);

        const auto mask1 = zeroAndAllOne[t_previous & u_previous];
        const auto mask2 = zeroAndAllOne[(t_previous ^ u_previous) & (t_previous | u_previous)];
        auto ss = s & notBytesBlock;

        block ct[2];
        AES ak(ss);
        static const block OneBlock = toBlock(0, 1);
        static const block ThreeBlock = toBlock(0, 3);
        block pt[2] = {OneBlock, ThreeBlock};
        ak.ecbEncTwoBlocks(pt, ct);

        if (t_previous == 0 && int(lsb(ct[0])) == 0){
            ct[0] = ct[0] & notTwoBlock;
        }

        s_right = ((scw_local ^ t2_bits_keep ^ (u2_bits_keep << 1)) & mask1)
                ^ ((scw_local ^ t1_bits_keep ^ (u1_bits_keep << 1)) & mask2)
                ^ ct[0];

        uint64_t v_converted = _mm_cvtsi128_si64x(ct[1]);
        v_converted &= 0x000000000FFFFFFFULL;

        uint64_t tmp = v_converted;
        GroupElement* vptr = (v + ((int)level) * 1);

        if (party == SERVER1){
            tmp = -2 * tmp;
        }
        if (t_previous == 1){
            if (party == SERVER1){
                tmp -= vptr->value;
            }
            else{
                uint64_t val = vptr->value;
                if(party == SERVER0 && (val & 1)){
                    val += 1;
                }
                if (val > (UINT64_MAX >> 1)){
                    val = (UINT64_MAX - ((UINT64_MAX - val)>>1));
                }
                else{
                    val >>= 1;
                }
                tmp += val;
            }
        }

        v_share_stack[level + 1].value = v_share_stack[level].value + tmp;
        traverseRangeSumDCFSerial_GS1(Bin, Bout, party, s_right, k0, k1, v, g, v_share_stack, sums, rangeStart, rangeLen, (pathIdx << 1) | 1, level + 1);
    }
}

void evalFullDomainRangeDCF(int party, GroupElement *res, const DCFKeyPack &key,
                            uint64_t start, uint64_t len, int splitDepth)
{
    // Aggregate instead of materializing every point.
    // Reuse the existing sum-oriented traversal to accumulate totals.
    evalFullDomainSumDCF(party, res, key, start, len, splitDepth);
}

void evalFullDomainSumDCF(int party, GroupElement *sums, const DCFKeyPack &key,
                            uint64_t start, uint64_t len, int splitDepth)
{
    if (len == 0) return;
    if (key.Bin < 0 || key.Bin >= 63) {
        return;
    }

    int actualSplitDepth = 20;
    if (splitDepth != -1) {
        actualSplitDepth = std::min(splitDepth, key.Bin);
    } else {
        actualSplitDepth = fss_recommended_split_depth(key.Bin, len);
    }

    for(int i=0; i<key.groupSize; ++i) sums[i].value = 0;

    if (!fss_should_parallelize_tree(key.Bin, actualSplitDepth, key.groupSize, len)) {
        std::vector<GroupElement> v_share_stack((key.Bin + 1) * key.groupSize, GroupElement(0, key.Bout));
        if (key.groupSize == 1) {
            traverseRangeSumDCFSerial_GS1(key.Bin, key.Bout, party,
                                          key.k0[0], key.k0, key.k1,
                                          key.v, key.g,
                                          v_share_stack.data(), sums,
                                          start, len, 0, 0);
        } else {
            traverseRangeSumDCFSerial(key.Bin, key.Bout, key.groupSize, party,
                                      key.k0[0], key.k0, key.k1,
                                      key.v, key.g,
                                      v_share_stack.data(), sums,
                                      start, len, 0, 0);
        }
        return;
    }

    int max_threads = omp_get_max_threads();
    std::vector<std::vector<DCFNode>> thread_nodes(max_threads * 8);
    for(int i=0; i<max_threads; ++i) thread_nodes[i*8].reserve(1024);

    std::vector<std::deque<GroupElement>> thread_data(max_threads * 8);

    std::vector<GroupElement> root_v_share(key.groupSize, GroupElement(0, key.Bout));

    #pragma omp parallel
    {
        #pragma omp single
        {
            collectDCFNodes(key.Bin, key.Bout, key.groupSize, party,
                        key.k0[0], key.k0, key.k1, key.v, root_v_share,
                        start, len, 0, 0, actualSplitDepth, thread_nodes, thread_data);
            #pragma omp taskwait
        }
    }

    std::vector<DCFNode> tasks;
    size_t total_tasks = 0;
    for(int i=0; i<max_threads; ++i) total_tasks += thread_nodes[i*8].size();
    tasks.reserve(total_tasks);
    for(int i=0; i<max_threads; ++i) tasks.insert(tasks.end(), thread_nodes[i*8].begin(), thread_nodes[i*8].end());

    if (!fss_should_parallelize_loop(tasks.size())) {
        std::vector<GroupElement> v_share_stack((key.Bin + 1) * key.groupSize, GroupElement(0, key.Bout));
        for (const auto &node : tasks) {
            GroupElement* current_level_ptr = &v_share_stack[node.level * key.groupSize];
            for (int g = 0; g < key.groupSize; ++g) {
                current_level_ptr[g] = node.v_share[g];
            }

            if (key.groupSize == 1) {
                traverseRangeSumDCFSerial_GS1(key.Bin, key.Bout, party,
                                              node.s, key.k0, key.k1,
                                              key.v, key.g,
                                              v_share_stack.data(), sums,
                                              start, len,
                                              node.path, node.level);
            } else {
                traverseRangeSumDCFSerial(key.Bin, key.Bout, key.groupSize, party,
                                          node.s, key.k0, key.k1,
                                          key.v, key.g,
                                          v_share_stack.data(), sums,
                                          start, len,
                                          node.path, node.level);
            }
        }
        return;
    }

    std::vector<std::vector<GroupElement>> thread_sums_vec(max_threads, std::vector<GroupElement>(key.groupSize, GroupElement(0, key.Bout)));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        GroupElement* local_sums = thread_sums_vec[tid].data();
        std::vector<GroupElement> v_share_stack((key.Bin + 1) * key.groupSize);

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < tasks.size(); ++i) {
            const auto& node = tasks[i];

            GroupElement* current_level_ptr = &v_share_stack[node.level * key.groupSize];
            for (int g = 0; g < key.groupSize; ++g) {
                current_level_ptr[g] = node.v_share[g];
            }

            if (key.groupSize == 1) {
                traverseRangeSumDCFSerial_GS1(key.Bin, key.Bout, party,
                                    node.s, key.k0, key.k1,
                                    key.v, key.g,
                                    v_share_stack.data(), local_sums,
                                    start, len,
                                    node.path, node.level);
            } else {
                traverseRangeSumDCFSerial(key.Bin, key.Bout, key.groupSize, party,
                                    node.s, key.k0, key.k1,
                                    key.v, key.g,
                                    v_share_stack.data(), local_sums,
                                    start, len,
                                    node.path, node.level);
            }
        }
    }
    for(int t=0; t<max_threads; ++t) {
        for(int i=0; i<key.groupSize; ++i) {
            sums[i].value += thread_sums_vec[t][i].value;
        }
    }
}
