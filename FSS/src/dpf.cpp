/*
Authors: Deepak Kumarasw// 定义函数 u 来获取倒数第二位 (bit 1)
inline u8 u(const block &b) {
    return (_mm_cvtsi128_si32(b) >> 1) & 1;
} Kanav Gupta
Copyright:
Copyright (c) 2022 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of thvoid void evalDvoid evalDPFPartial(int party, GroupElement *res, GroupElement idx, const DCFKeyPack &key, int start, int len) party, GroupElement *res, GroupElement idx, const DPFKeyPack &key)
{
    evalDPF(key.Bin, key.Bout, key.groupSize, res, party, idx, key.k0, key.g);
}

void evalDPFPartial(int party, GroupElement *res, GroupElement idx, const DCFKeyPack &key, int start, int len) party, GroupElement *res, GroupElement idx, const DPFKeyPack &key)
{
    evalDPF(key.Bin, key.Bout, key.groupSize, res, party, idx, key.k0, key.g);
}

void evalDPFPartial(int party, GroupElement *res, GroupElement idx, const DCFKeyPack &key, int start, int len) and to permit persons to whom the Software is
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

#include "dpf.h"
#include <bitset>
#include <omp.h>
#include <array>
#include <cstdint>

using namespace osuCrypto;

std::array<block, 2> Hash(uint64_t input, block s) {
    std::array<block, 2> res;
    AES aes(s);
    block pt[2];
    pt[0] = toBlock(input, 0);
    pt[1] = toBlock(input, 1);
    aes.ecbEncTwoBlocks(pt, res.data());
    return res;
}


// uint64_t aes_evals_count = 0;

// // 定义函数 u 来获取倒数第二位 (bit 1)
// inline u8 u(const block &b) {
//     return (_mm_cvtsi128_si32(b) >> 1) & 1;
// }

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

#ifndef FSS_clearAESevals_DEFINED
#define FSS_clearAESevals_DEFINED
void clearAESevals()
{
    // aes_evals_count = 0;
}
#endif

#ifndef FSS_bytesize_DEFINED
#define FSS_bytesize_DEFINED
inline int bytesize(const int bitsize) {
    return (bitsize % 8) == 0 ? bitsize / 8 : (bitsize / 8)  + 1;
}
#endif

#ifndef FSS_convert_DEFINED
#define FSS_convert_DEFINED
void convert(const int bitsize, const int groupSize, const block &b, uint64_t *out, bool forceParallel)
{
    static const block notThreeBlock = toBlock(~0, ~3);
    const int bys = bytesize(bitsize);
    const int totalBys = bys * groupSize;
    if (bys * groupSize <= 16) {
        uint8_t *bptr = (uint8_t *)&b;
#pragma omp parallel for if(fss_should_parallelize_loop(groupSize, forceParallel))
        for(int i = 0; i < groupSize; i++) {
            out[i] = *(uint64_t *)(bptr + i * bys);
        }
    }
    else {
        int numblocks = totalBys % 16 == 0 ? totalBys / 16 : (totalBys / 16) + 1;
        AES aes(b);
        block pt[numblocks];
        block ct[numblocks];
#pragma omp parallel for if(fss_should_parallelize_loop(numblocks, forceParallel))
        for(int i = 0; i < numblocks; i++) {
            pt[i] = toBlock(0, i);
        }
        aes.ecbEncBlocks(pt, numblocks, ct);
        uint8_t *bptr = (uint8_t *)ct;
#pragma omp parallel for if(fss_should_parallelize_loop(groupSize, forceParallel))
        for(int i = 0; i < groupSize; i++) {
            out[i] = *(uint64_t *)(bptr + i * bys);
        }
    }
}

void convert(const int bitsize, const int groupSize, const block &b, uint64_t *out)
{
    convert(bitsize, groupSize, b, out, false);
}
#endif

block traverseOneDPF(int Bin, int Bout, int groupSize, int party,
                        block &s,
                        const block &cw1,
                        const block &cw2,
                        const u8 &keep,
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
    block ct[1]; // {tau, v_this_level}  存储当前s值和v值
    u8 t_previous = lsb(s);
    // if(t_previous == 0){
    //     s = s & notTwoBlock;  // 将倒数第二位强制设置为0
    // }
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
    ak.ecbEncBlock(blocks[2 * keep], ct[0]); // 修改为只加密一个块
    if (t_previous ==0 && int(lsb(ct[0])) == 0){
        ct[0] = ct[0] & notTwoBlock;  // 将倒数二位强制设置为0
    }
    // std::cout<<"当前层数: "<<level<<"  party: "<<party<<
    //             "  t_previous: "<<(int)t_previous << "  u_previous: "<<(int)u_previous
    //             <<"  t_now: "<<int(lsb(ct[0]))<<"  u_now: "<<int(lsb(ct[0]>>1))<<std::endl;
    stcw = ((scw ^ t2_bits[keep] ^ (u2_bits[keep] << 1)) & mask1)
            ^ ((scw ^ t1_bits[keep] ^ (u1_bits[keep] << 1)) & mask2)
            ^ ct[0];
    // std::cout<<"当前层数: "<<level<<"  party: "<<party<<"  keep: "<<(int)keep
    //             <<"  t—: "<<int(lsb(stcw))
    //             <<"  u—: "<<int(lsb(stcw>>1))
    //             <<std::endl;
    // if(t_previous & u_previous){
    //     std::cout<<"使用cw2:    "<<t2_bits[keep]<<" "<<u2_bits[keep] <<std::endl;
    // }
    // else if ((t_previous ^ u_previous) & (t_previous | u_previous)){
    //     std::cout<<"使用cw1:    "<<t1_bits[keep]<<" "<<u1_bits[keep] <<std::endl;
    // }
    // else{
    //     std::cout<<"不使用cw:    00 00"<<std::endl;
    // }
    // std::cout<<"加密结果:    "<<int(lsb(stcw))<<" "<<int(lsb(stcw>>1))<<std::endl;
    return stcw;
}


block traversePathDPF(int Bin, int Bout, int groupSize, int party,
                        GroupElement idx,
                        block *k0,
                        block *k1,
                        bool geq,
                        int evalGroupIdxStart,
                        int evalGroupIdxLen)
{
    block s = k0[0];
    for (int i = 0; i < Bin; ++i)
    {
        const u8 keep = static_cast<uint8_t>(idx.value >> (Bin - 1 - i)) & 1;
        s = traverseOneDPF(Bin, Bout, groupSize, party, s, k0[i + 1], k1[i + 1],keep, i, geq, evalGroupIdxStart, evalGroupIdxLen);
    }
    return s;
}


// Real Endpoints
std::tuple<DPFKeyPack, DPFKeyPack, DPFKeyPack> keyGenDPF(int Bin, int Bout, int groupSize,
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
    const static block pt[2] = {ZeroBlock, OneBlock};

    //三方DCF修改
    auto s = prng.get<std::array<block, 3>>();  //创建初始三个结点
    block si[3][2];
    //


    //三方DCF修改
    block *k0 = new block[Bin + 1];
    block *k1 = new block[Bin + 1];
    block *k2 = new block[Bin + 1];
    block *k_ = new block[Bin + 1];
    //
    GroupElement *v0 = new GroupElement[Bin * groupSize];
    // 创建二维数组：groupSize行，3列 (使用一维数组模拟)
    GroupElement* g0 = new GroupElement[groupSize];

    //三方DCF修改
    s[0] = (s[0] & notOneBlock) ^ (s[1] & OneBlock) ^ OneBlock;  //使得t0^t1=为1
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
    block ct[2];
    //

    // std::cout<<"初始s0: "<<_mm_cvtsi128_si64x(s[0])<<"  s1: "<<_mm_cvtsi128_si64x(s[1])<<"  s2: "<<_mm_cvtsi128_si64x(s[2])<<std::endl;
    // // 输出 s[0], s[1], s[2] 的二进制表示
    // for (int i = 0; i < 3; ++i) {
    //     std::cout << "s[" << i << "] 二进制: ";
    //     uint8_t *bytes = (uint8_t*)&s[i];
    //     for (int j = 15; j >= 0; --j) {  // 从高字节到低字节
    //         std::cout << std::bitset<8>(bytes[j]);
    //     }
    //     std::cout << std::endl;
    // }
    for (int i = 0; i < Bin; ++i)
    {
        const u8 keep = static_cast<uint8_t>(idx.value >> (Bin - 1 - i)) & 1;
        auto a = toBlock(keep);

        //三方DPF修改
        auto ss0 = s[0] & notBytesBlock;
        auto ss1 = s[1] & notBytesBlock;
        auto ss2 = s[2] & notBytesBlock;
        //

        AES ak0(ss0);
        AES ak1(ss1);
        AES ak2(ss2);
        ak0.ecbEncTwoBlocks(pt, ct);
        si[0][0] = ct[0];
        si[0][1] = ct[1];
        ak1.ecbEncTwoBlocks(pt, ct);
        si[1][0] = ct[0];
        si[1][1] = ct[1];
        ak2.ecbEncTwoBlocks(pt, ct);
        si[2][0] = ct[0];
        si[2][1] = ct[1];

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
        GroupElement sign(-1, Bout);


        //创建两套纠正词，找到0位
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

        // if (i == Bin - 1) {
        //     //最后一层
        //     std::cout <<"第" <<i <<"层" << "    "<< " "<< " "<< "zeroIdx: " << zeroIdx << std::endl;
        //     std::cout <<"第" <<i <<"层" << "    "<< " "<< ""<< "siXOR1: " << _mm_cvtsi128_si64x(siXOR1[0]) <<"     "<< "siXOR1: " << _mm_cvtsi128_si64x(siXOR1[1])<< std::endl;
        //     std::cout <<"第" <<i <<"层" << "    "<< " "<< ""<< "siXOR2: " << _mm_cvtsi128_si64x(siXOR2[0]) <<"     "<< "siXOR2: " << _mm_cvtsi128_si64x(siXOR2[1])<< std::endl;
        // }

        // check lose control bit
        // std::cout <<"第" <<i <<"层" << "    "<< " "<< " "<< "zeroIdx: " << zeroIdx << std::endl;
        // std::cout <<"第" <<i <<"层" << "    "<< " "<< " "<< "siXOR1: " << siXOR1[0] <<"     "<< "siXOR1: " << siXOR1[1]<< std::endl;

        // get the left and right t_CW bits,and check t，避免α路径上出现两个t=0的情况
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

        // std::cout <<"第" <<i <<"层" << "    "<< " "<< " "<<"s1_lose: " << int(lsb(siXOR1[keep^1])) <<"   "<<"s_lose: "<< int(lsb(siXOR1[keep^1]>>1))<< std::endl;
        // std::cout <<"第" <<i <<"层" << "    "<< " "<< " "<<"tcw_lose: " << _mm_cvtsi128_si64x(t1[keep^1]) <<"   "<<"ucw_lose: "<< _mm_cvtsi128_si64x(u1[keep^1])<< std::endl;
        // std::cout <<"第" <<i <<"层" << "    "<< " "<< " "<<"s2_lose: " << int(lsb(siXOR2[keep^1])) <<"   "<<"s2_lose: "<< int(lsb(siXOR2[keep^1]>>1))<< std::endl;
        // std::cout <<"第" <<i <<"层" << "    "<< " "<< " "<<"t2cw_lose: " << _mm_cvtsi128_si64x(t2[keep^1]) <<"   "<<"u2cw_lose: "<< _mm_cvtsi128_si64x(u2[keep^1])<< std::endl;

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
        // if (int(lsb(si[zeroIdx][keep]>>1)) == 1  && int(lsb(si[zeroIdx][keep])) == 1){
        //     u1[keep] = (si[one_party][keep] & TwoBlock) >> 1;
        //     u2[keep] = (si[both_one_party][keep] & TwoBlock) >> 1;
        // }
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

        //check scw every layer
        // std::cout <<"第" <<i <<"层" << "    "<< " "<< " "<<"scw1: " << _mm_cvtsi128_si64x(scw1) << std::endl;
        // std::cout <<"第" <<i <<"层" << "    "<< " "<< " "<<"scw2: " << _mm_cvtsi128_si64x(scw2) << std::endl;
        // std::cout <<"第" <<i <<"层" << "    "<< " "<< " "<<"t1_keep: " << _mm_cvtsi128_si64x(t1[keep]) << "    "<< "t1_lose:    "<<_mm_cvtsi128_si64x(t1[keep ^ 1]) << std::endl;
        // std::cout <<"第" <<i <<"层" << "    "<< " "<< " "<<"t2_keep: " << _mm_cvtsi128_si64x(t2[keep]) << "    "<< "t2_lose:    "<<_mm_cvtsi128_si64x(t2[keep ^ 1]) << std::endl;
        // std::cout <<"第" <<i <<"层" << "    "<< " "<< " "<<"u1_keep: " << _mm_cvtsi128_si64x(u1[keep]) << "    "<< "u1_lose:    "<<_mm_cvtsi128_si64x(u1[keep ^ 1]) << std::endl;
        // std::cout <<"第" <<i <<"层" << "    "<< " "<< ""<<"u2_keep: " << _mm_cvtsi128_si64x(u2[keep]) << "    "<< "u2_lose: "<<_mm_cvtsi128_si64x(u2[keep ^ 1]) << std::endl;


        // k0[i + 1] = k1[i + 1] = k1[i + 1] = scw1           // set bits [127, 2] as scw = s0_loss ^ s1_loss
        //                     ^ (t1[0] << 1) // set bit as tL
        //                     ^ t1[1]        // set bit as tR
        //                     ^ (t2[0] << 3) // set bit as t2L
        //                     ^ (t2[1] << 2)       // set bit as t2R
        //                     ^ (u1[0] << 5) // set bit as uL
        //                     ^ (u1[1] << 4)       // set bit 0 as uR
        //                     ^ (u2[0] << 7) // set bit as u2L
        //                     ^ (u2[1] << 6);       // set bit  as u2R

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
        // std::cout <<"第" <<i <<"层" << "    "<< " "<< " "<<"zeroIdx: " << zeroIdx << std::endl;
        // std::cout <<"第" <<i <<"层" << "    "<< " "<< " "<<"oneIdx: " << one_party << std::endl;
        // std::cout <<"第" <<i <<"层" << "    "<< " "<< " "<<"bothIdx: " << both_one_party << std::endl;
        // std::cout <<"第" <<i <<"层" << "    "<< " "<< " "<<"TKeep1: " << static_cast<int>(lsb(TKeep1)) << "    "<< "UKeep1: "<< static_cast<int>(lsb(UKeep1>>1)) << std::endl;
        // std::cout <<"第" <<i <<"层" << "    "<< " "<< " " << "TKeep2: " << static_cast<int>(lsb(TKeep2)) << "    "<< "UKeep2:   " << static_cast<int>(lsb(UKeep2>>1)) << std::endl;


        s[zeroIdx] = si0Keep ^ (zeroAndAllOne[t_bits[zeroIdx] & u_bits[zeroIdx]] & (scw2 ^ TKeep2 ^ UKeep2))
                            ^ (zeroAndAllOne[t_bits[zeroIdx] ^ u_bits[zeroIdx]] & (scw1 ^ TKeep1 ^ UKeep1));

        s[one_party] = si1Keep ^ (zeroAndAllOne[t_bits[one_party] & u_bits[one_party]] & (scw2 ^ TKeep2 ^ UKeep2))
                            ^ (zeroAndAllOne[t_bits[one_party] ^ u_bits[one_party]] & (scw1 ^ TKeep1 ^ UKeep1));
        s[both_one_party] = si2Keep ^ (zeroAndAllOne[t_bits[both_one_party] & u_bits[both_one_party]] & (scw2 ^ TKeep2 ^ UKeep2))
                            ^ (zeroAndAllOne[t_bits[both_one_party] ^ u_bits[both_one_party]] & (scw1 ^ TKeep1 ^ UKeep1));

        // std::cout <<"第" <<i <<"层" << "    "<< " "<< " "<<"t1: " << static_cast<int>(lsb(s[0])) << "    "<< "u1_keep:  "<< static_cast<int>(lsb(s[0]>>1)) << std::endl;
        // std::cout <<"第" <<i <<"层" << "    "<< " "<< " "<<"t2: " << static_cast<int>(lsb(s[1]))<< "    "<< "u2_keep:   " << static_cast<int>(lsb(s[1]>>1)) << std::endl;
        // std::cout <<"第" <<i <<"层" << "    "<< " "<< " "<<"t3: " << static_cast<int>(lsb(s[2])) << "    "<< "u3_keep:  "<< static_cast<int>(lsb(s[2]>>1)) << std::endl;
    }
    //check value of the last layer
    // std::cout << "Final s0: " << _mm_cvtsi128_si64x(s[0]) << std::endl;
    // std::cout << "Final s1: " << _mm_cvtsi128_si64x(s[1]) << std::endl;
    // std::cout << "Final s2: " << _mm_cvtsi128_si64x(s[2]) << std::endl;

    uint64_t s0_converted[groupSize];
    uint64_t s1_converted[groupSize];
    uint64_t s2_converted[groupSize];
    convert(Bout, groupSize, s[0] & notBytesBlock, s0_converted);
    convert(Bout, groupSize, s[1] & notBytesBlock, s1_converted);
    convert(Bout, groupSize, s[2] & notBytesBlock, s2_converted);

    #pragma omp parallel for if(fss_should_parallelize_loop(groupSize))
    for (int lp = 0; lp < groupSize; ++lp)
    {
        s0_converted[lp] &= 0x000000000FFFFFFFULL;
        s1_converted[lp] &= 0x000000000FFFFFFFULL;
        s2_converted[lp] &= 0x000000000FFFFFFFULL;
        // 当t1=1时,cwn+1需要加上payload，并对s1的值修改 cw=2s1-s0-s2+2payload
        if (lsb(s[1]) == 1){
            g0[lp] =  (s0_converted[lp] + s2_converted[lp] - (payload[lp].value * 2ULL) - (s1_converted[lp] * 2ULL));
            // if (s0_converted[lp] > UINT64_MAX - s2_converted[lp]){
            //     g0[lp] =  UINT64_MAX - (UINT64_MAX - g0[lp]);
            // }
            // std::cout << "s0: " << s0_converted[lp]<< " s1: " << s1_converted[lp]<< " s2: " << s2_converted[lp]<< std::endl;
            // std::cout << "值1: " << s0_converted[lp] + s2_converted[lp]<< std::endl;
            // std::cout << "值2: " << (s0_converted[lp] + s2_converted[lp] - (payload[lp].value * 2ULL))<< std::endl;
            // std::cout << "值3: " << g0[lp].value<< std::endl;
        }
        else{
            // 当t1=0时,cwn+1需要加上payload，并对s1的值修改 cw=s1-s0/2-s2/2+payload
            g0[lp] = (s1_converted[lp]  + payload[lp].value - (s0_converted[lp]>>1) - (s2_converted[lp] >> 1));
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

    return std::make_tuple(DPFKeyPack(Bin, Bout, groupSize, k0, k_, g0, cs), DPFKeyPack(Bin, Bout, groupSize, k1, k_, g0, cs), DPFKeyPack(Bin, Bout, groupSize, k2, k_, g0, cs));
}

// std::tuple<DCFKeyPack, DCFKeyPack, DCFKeyPack> keyGenDPF(int Bin, int Bout,
//                 GroupElement idx, GroupElement payload)
// {
//     // idx: bitsize Bin, payload: bitsize Bout
//     return keyGenDPF(Bin, Bout, 1, idx, &payload);
// }

std::tuple<DPFKeyPack, DPFKeyPack,DPFKeyPack> keyGenDPF(int Bin, int Bout,
                GroupElement idx, GroupElement payload, const GenParams &gen)
{
    // 原来的两方DCF实现 - 需要重新实现或调用原来的逻辑
    // 这里暂时返回空的pair，需要根据原来的两方DCF逻辑来实现
    auto result = keyGenDPF(Bin, Bout, 1, idx, &payload, gen);
    return std::make_tuple(std::get<0>(result), std::get<1>(result),std::get<2>(result));
}

void evalDPF(int Bin, int Bout, int groupSize,
                GroupElement *out, // groupSize
                int party, GroupElement idx,
                block *k0, // bin + 1
                block *k1,
                GroupElement *g , // groupSize
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

    auto s = traversePathDPF(Bin, Bout, groupSize, party, idx, k0,k1, geq, evalGroupIdxStart, evalGroupIdxLen);

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

    uint64_t s_converted[groupSize];
    // uint64_t tmp1 = 4077616351633564717;
    // uint64_t tmp2 = 14369127719947780691;
    // uint64_t tmp3 = 3072661296;
    // std::cout << "tmp: " << tmp1/2-tmp2+tmp3/2 << std::endl;
    static const block notThreeBlock = toBlock(~0, ~3);
    static const block BytesBlock = toBlock(0, 15);
    static const block notBytesBlock = toBlock(~0, ~15);
    convert(Bout, groupSize, s & notBytesBlock, s_converted);
    // std::cout<<"  s: "<<s_converted[0]<<std::endl;
    //聚合的时候计算0.5*s0-s1+0.5*s2
    #pragma omp parallel for if(fss_should_parallelize_loop(evalGroupIdxLen))
    for(int ctr = 0; ctr < evalGroupIdxLen; ++ctr)
    {
        int lp = (evalGroupIdxStart + ctr) % groupSize;
        s_converted[lp] &= 0x000000000FFFFFFFULL;
        // uint64_t sign = (party == SERVER1)?1:0;
        GroupElement final_term = s_converted[lp];
        // std::cout << "Final s block: " << final_term.value << std::endl;
        // std::cout << "t: " << int(t) << std::endl;
        if (t == 1)
        {
            final_term.value = final_term.value + g[lp].value;
        }
        if (party == SERVER1){final_term.value = -1 * final_term.value;}
        else{
            // std::cout<< "Max: "<<UINT64_MAX<<std::endl;
            if (party == SERVER0 && (final_term.value & 1 == 1)){
                final_term.value += 1;
            }
            if (final_term.value > (UINT64_MAX >> 1)){
                final_term.value = (UINT64_MAX - ((UINT64_MAX - final_term.value)>>1));
            }
            else{
                final_term.value >>= 1;
            }

        }
        out[lp].value = out[lp].value + final_term.value;
        // std::cout <<" T : "<<int(t)<<"   "<< "Final s block: " << final_term.value << std::endl;
        // std::cout<<"eval  party: "<<party<<"  s: "<<s_converted[lp]<<"  g:  "<<g[lp].value<< std::endl<< std::endl;
    }
}

void evalDPF(int party, GroupElement *res, GroupElement idx, const DPFKeyPack &key, bool enableVerify)
{
    evalDPF(key.Bin, key.Bout, key.groupSize, res, party, idx, key.k0, key.k1, key.g, false, 0, -1, enableVerify, key.cs);
}


void evalDPFPartial(int party, GroupElement *res, GroupElement idx, const DPFKeyPack &key, int start, int len)
{
    evalDPF(key.Bin, key.Bout, key.groupSize, res, party, idx, key.k0, key.k1,key.g, false, start, len);
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
    // Heuristic goals:
    // 1) For Bin>20, keep subtree depth <= ~20 when possible.
    // 2) Ensure enough tasks to saturate cores (D>=10 => up to 1024 tasks).
    // 3) Cap task count to avoid excessive overhead (D<=16 => <=65536 tasks).
    int d = std::max(0, Bin - 20);
    d = std::max(d, 10);
    d = std::min(d, 16);
    d = std::min(d, Bin);

    if (rangeLen > 0) {
        int r = fss_floor_log2_u64(rangeLen);
        if (r >= 0) {
            // Try to make each task's subtree size roughly <= rangeLen.
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

// Aggregate-only traversal: accumulate values instead of materializing every point.
static void traverseRangeDPFSerial(int Bin, int Bout, int groupSize, int party,
                                   block s,
                                   block *k0, block *k1, GroupElement *g,
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
        (void)leafIdx; // leaf index only used for range checks above
        u8 t = lsb(s);
        static const block notBytesBlock = toBlock(~0, ~15);

        // Optimized for GroupSize=1
        if (groupSize == 1) {
            // Inline convert and logic
             block s_masked = s & notBytesBlock;
             uint64_t val = _mm_cvtsi128_si64x(s_masked);
             val &= 0x000000000FFFFFFFULL;

             GroupElement final_term(val, Bout);
             if (t == 1) {
                final_term.value += g[0].value;
             }

             if (party == SERVER1){
                final_term.value = (uint64_t)(-1 * (int64_t)final_term.value);
             }
             else{
                if (party == SERVER0 && ((final_term.value & 1) == 1)){
                    final_term.value += 1;
                }
                if (final_term.value > (UINT64_MAX >> 1)){
                    final_term.value = (UINT64_MAX - ((UINT64_MAX - final_term.value)>>1));
                }
                else{
                    final_term.value >>= 1;
                }
             }
               sums[0].value += final_term.value;
             return;
        }

        uint64_t s_converted[groupSize];
        convert(Bout, groupSize, s & notBytesBlock, s_converted);

#pragma omp parallel for if(!omp_in_parallel() && groupSize >= 1024)
        for(int lp = 0; lp < groupSize; ++lp) {
            s_converted[lp] &= 0x000000000FFFFFFFULL;
            GroupElement final_term(s_converted[lp], Bout);
            if (t == 1) {
                final_term.value = final_term.value + g[lp].value;
            }
            if (party == SERVER1){
                final_term.value = (uint64_t)(-1 * (int64_t)final_term.value);
            }
            else{
                if (party == SERVER0 && ((final_term.value & 1) == 1)){
                    final_term.value += 1;
                }
                if (final_term.value > (UINT64_MAX >> 1)){
                    final_term.value = (UINT64_MAX - ((UINT64_MAX - final_term.value)>>1));
                }
                else{
                    final_term.value >>= 1;
                }
            }
            sums[lp].value += final_term.value;
        }
        return;
    }

    // Optimization: Inline traverseOneDPF and share AES expansion
    static const block notTwoBlock = toBlock(~0, ~2);
    static const block OneBlock = toBlock(0, 1);
    static const block notBytesBlock = toBlock(~0, ~15);
    static const block zeroAndAllOne[2] = {ZeroBlock, toBlock(~0, ~0)};

    block cw1 = k0[level + 1];
    block cw2 = k1[level + 1];

    u8 t_previous = lsb(s);
    u8 u_previous = lsb(s>>1);

    auto scw = (cw1 & notBytesBlock);
    if (t_previous & u_previous){
        scw = (cw2 & notBytesBlock);
    }

    auto ss = s & notBytesBlock;
    block ct[2];
    block inputs_enc[2] = {ZeroBlock, OneBlock};
    AES ak(ss);
    ak.ecbEncBlocks(inputs_enc, 2, ct);

    const auto mask1 = zeroAndAllOne[t_previous & u_previous];
    const auto mask2 = zeroAndAllOne[(t_previous ^ u_previous) & (t_previous | u_previous)];

    block t1_bits_0 = ((cw1 >> 1) & OneBlock);
    block t1_bits_1 = (cw1 & OneBlock);

    block t2_bits_0 = ((cw2 >> 1) & OneBlock);
    block t2_bits_1 = (cw2 & OneBlock);

    block u1_bits_0 = ((cw1 >> 3) & OneBlock);
    block u1_bits_1 = ((cw1 >> 2) & OneBlock);

    block u2_bits_0 = ((cw2 >> 3) & OneBlock);
    block u2_bits_1 = ((cw2 >> 2) & OneBlock);

    // LEFT (keep=0)
    {
        block stcw;
        block ct_val = ct[0];
        if (t_previous == 0 && int(lsb(ct_val)) == 0){
             ct_val = ct_val & notTwoBlock;
        }
        stcw = ((scw ^ t2_bits_0 ^ (u2_bits_0 << 1)) & mask1)
             ^ ((scw ^ t1_bits_0 ^ (u1_bits_0 << 1)) & mask2)
             ^ ct_val;

        traverseRangeDPFSerial(Bin, Bout, groupSize, party, stcw, k0, k1, g, sums, rangeStart, rangeLen, pathIdx << 1, level + 1);
    }

    // RIGHT (keep=1)
    {
        block stcw;
        block ct_val = ct[1];
        if (t_previous == 0 && int(lsb(ct_val)) == 0){
             ct_val = ct_val & notTwoBlock;
        }
        stcw = ((scw ^ t2_bits_1 ^ (u2_bits_1 << 1)) & mask1)
             ^ ((scw ^ t1_bits_1 ^ (u1_bits_1 << 1)) & mask2)
             ^ ct_val;

        traverseRangeDPFSerial(Bin, Bout, groupSize, party, stcw, k0, k1, g, sums, rangeStart, rangeLen, (pathIdx << 1) | 1, level + 1);
    }
}

struct DPFNode {
    block s;
    uint64_t path;
    int level;
};

static void collectDPFNodesSerial(int Bin, int Bout, int groupSize, int party,
                            block s, block *k0, block *k1,
                            uint64_t rangeStart, uint64_t rangeLen,
                            uint64_t pathIdx, int level, int splitDepth,
                            std::vector<std::vector<DPFNode>>& thread_nodes)
{
    // Pruning
    const uint64_t subtreeLen = fss_pow2_u64(Bin - level);
    const uint64_t subtreeStart = pathIdx << (Bin - level);

    if (subtreeLen == 0 || fss_range_disjoint(subtreeStart, subtreeLen, rangeStart, rangeLen)) {
        return;
    }

    if (level == splitDepth || level == Bin) {
        int tid = omp_get_thread_num();
        thread_nodes[tid * 8].push_back({s, pathIdx, level});
        return;
    }

    // Recurse Left
    block s_left = s;
    s_left = traverseOneDPF(Bin, Bout, groupSize, party, s_left, k0[level + 1], k1[level + 1], 0, level, false, 0, -1, false);
    collectDPFNodesSerial(Bin, Bout, groupSize, party, s_left, k0, k1, rangeStart, rangeLen, pathIdx << 1, level + 1, splitDepth, thread_nodes);

    // Recurse Right
    block s_right = s;
    s_right = traverseOneDPF(Bin, Bout, groupSize, party, s_right, k0[level + 1], k1[level + 1], 1, level, false, 0, -1, false);
    collectDPFNodesSerial(Bin, Bout, groupSize, party, s_right, k0, k1, rangeStart, rangeLen, (pathIdx << 1) | 1, level + 1, splitDepth, thread_nodes);
}

static void collectDPFNodes(int Bin, int Bout, int groupSize, int party,
                            block s, block *k0, block *k1,
                            uint64_t rangeStart, uint64_t rangeLen,
                            uint64_t pathIdx, int level, int splitDepth,
                            std::vector<std::vector<DPFNode>>& thread_nodes)
{
    // Pruning
    const uint64_t subtreeLen = fss_pow2_u64(Bin - level);
    const uint64_t subtreeStart = pathIdx << (Bin - level);

    if (subtreeLen == 0 || fss_range_disjoint(subtreeStart, subtreeLen, rangeStart, rangeLen)) {
        return;
    }

    // Switch to Serial if depth is deep enough
    if (level >= splitDepth - 4) {
        collectDPFNodesSerial(Bin, Bout, groupSize, party, s, k0, k1, rangeStart, rangeLen, pathIdx, level, splitDepth, thread_nodes);
        return;
    }

    if (level == splitDepth || level == Bin) {
        int tid = omp_get_thread_num();
        thread_nodes[tid * 8].push_back({s, pathIdx, level});
        return;
    }

    // Recurse Left
    #pragma omp task shared(thread_nodes) if(level < splitDepth - 4)
    {
        block s_left = s;
        // forceParallel = true
        s_left = traverseOneDPF(Bin, Bout, groupSize, party, s_left, k0[level + 1], k1[level + 1], 0, level, false, 0, -1, true);
        collectDPFNodes(Bin, Bout, groupSize, party, s_left, k0, k1, rangeStart, rangeLen, pathIdx << 1, level + 1, splitDepth, thread_nodes);
    }

    // Recurse Right
    #pragma omp task shared(thread_nodes) if(level < splitDepth - 4)
    {
        block s_right = s;
        s_right = traverseOneDPF(Bin, Bout, groupSize, party, s_right, k0[level + 1], k1[level + 1], 1, level, false, 0, -1, true);
        collectDPFNodes(Bin, Bout, groupSize, party, s_right, k0, k1, rangeStart, rangeLen, (pathIdx << 1) | 1, level + 1, splitDepth, thread_nodes);
    }
    #pragma omp taskwait
}

// Aggregate over [start, start+len) and return the sum per group element in res (size=groupSize).
// This no longer materializes every point; only totals are produced.
void evalFullDomainRangeDPF(int party, GroupElement *res, const DPFKeyPack &key,
                            uint64_t start, uint64_t len, int splitDepth)
{
    if (len == 0) return;
    if (key.Bin < 0 || key.Bin >= 63) {
        return;
    }

    for(int g = 0; g < key.groupSize; ++g) {
        res[g].value = 0;
    }

    int actualSplitDepth = 20;
    if (splitDepth != -1) {
        actualSplitDepth = std::min(splitDepth, key.Bin);
    } else {
        actualSplitDepth = fss_recommended_split_depth(key.Bin, len);
    }

    if (!fss_should_parallelize_tree(key.Bin, actualSplitDepth, key.groupSize, len)) {
        traverseRangeDPFSerial(key.Bin, key.Bout, key.groupSize, party,
                               key.k0[0], key.k0, key.k1, key.g,
                               res, start, len, 0, 0);
        return;
    }

    // Thread-local storage for nodes to avoid critical section
    // Use padding (x8) to avoid false sharing on vector metadata
    int max_threads = omp_get_max_threads();
    std::vector<std::vector<DPFNode>> thread_nodes(max_threads * 8);
    // Estimate reservation to avoid reallocations
    for(int i=0; i<max_threads; ++i) thread_nodes[i*8].reserve(4096);

    block rootS = key.k0[0];

    // 1. Recursive collection phase
    #pragma omp parallel
    {
        #pragma omp single
        {
            collectDPFNodes(key.Bin, key.Bout, key.groupSize, party,
                        rootS, key.k0, key.k1,
                        start, len,
                        0, 0, actualSplitDepth, thread_nodes);
            #pragma omp taskwait
        }
    }

    // 2. Flatten nodes
    std::vector<DPFNode> nodes;
    size_t total_nodes = 0;
    for(int i=0; i<max_threads; ++i) total_nodes += thread_nodes[i*8].size();

    nodes.reserve(total_nodes);
    for(int i=0; i<max_threads; ++i) nodes.insert(nodes.end(), thread_nodes[i*8].begin(), thread_nodes[i*8].end());

    if (!fss_should_parallelize_loop(nodes.size())) {
        for (const auto &node : nodes) {
            traverseRangeDPFSerial(key.Bin, key.Bout, key.groupSize, party,
                                   node.s, key.k0, key.k1, key.g,
                                   res, start, len,
                                   node.path, node.level);
        }
        return;
    }

    // 3. Parallel Execution with per-thread aggregates
    std::vector<std::vector<GroupElement>> thread_sums_vec(max_threads, std::vector<GroupElement>(key.groupSize, GroupElement(0, key.Bout)));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        GroupElement* local_sums = thread_sums_vec[tid].data();

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < nodes.size(); ++i) {
            const auto& node = nodes[i];
            traverseRangeDPFSerial(key.Bin, key.Bout, key.groupSize, party,
                                   node.s, key.k0, key.k1, key.g,
                                   local_sums, start, len,
                                   node.path, node.level);
        }
    }

    // 4. Reduce thread-local sums
    for(int t=0; t<max_threads; ++t) {
        for(int g=0; g<key.groupSize; ++g) {
            res[g].value += thread_sums_vec[t][g].value;
        }
    }
}

void evalFullDomainDPF(int party, GroupElement *res, const DPFKeyPack &key)
{
    const uint64_t domain = fss_pow2_u64(key.Bin);
    evalFullDomainRangeDPF(party, res, key, 0, domain, -1);
}
