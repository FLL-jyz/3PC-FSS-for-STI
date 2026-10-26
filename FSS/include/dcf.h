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

#pragma once
#include <array>
#include <vector>
#include <utility>
#include <cryptoTools/Common/Defines.h>
#include <cryptoTools/Crypto/AES.h>
#include <cryptoTools/Crypto/PRNG.h>
#include <cryptoTools/gsl/span>
#include "group_element.h"
#include "keypack.h"

using namespace osuCrypto;

// extern uint64_t aes_evals_count;

void clearAESevals();
#ifndef FSS_LSB_DEFINED
#define FSS_LSB_DEFINED
inline u8 lsb(const block &b)
{
    return _mm_cvtsi128_si64x(b) & 1;
}
#endif

// Forward declarations for utility functions defined in dpf.cpp
void convert(const int bitsize, const int groupSize, const block &b, uint64_t *out, bool forceParallel);
void convert(const int bitsize, const int groupSize, const block &b, uint64_t *out);
std::array<block, 2> Hash(uint64_t input, block s);
inline int bytesize(const int bitsize);

std::tuple<DCFKeyPack, DCFKeyPack,DCFKeyPack> keyGenDCF(int Bin, int Bout, int groupSize,
                GroupElement idx, GroupElement* payload, const GenParams &gen = GenParams());

std::tuple<DCFKeyPack, DCFKeyPack,DCFKeyPack> keyGenDCF(int Bin, int Bout,
                GroupElement idx, GroupElement payload, const GenParams &gen = GenParams());

void evalDCF(int party, GroupElement *res, GroupElement idx, const DCFKeyPack &key, bool enableVerify = false);
void evalDCF(int Bin, int Bout, int groupSize,
                GroupElement *out, // groupSize
                int party, GroupElement idx,
                block *k0, // bin + 1
                block *k1,
                GroupElement *g , // groupSize
                GroupElement *v, // bin * groupSize
                bool geq = false, int evalGroupIdxStart = 0,
                int evalGroupIdxLen = -1,
                bool enableVerify = false,
                std::array<std::array<block, 2>, 2> cs = {{{ZeroBlock, ZeroBlock}, {ZeroBlock, ZeroBlock}}});

void evalDCFPartial(int party, GroupElement *res, GroupElement idx, const DCFKeyPack &key, int start, int len);

void evalFullDomainDCF(int party, GroupElement *res, const DCFKeyPack &key);

// Evaluate a contiguous range [start, start+len) of the full domain (len entries).
// Output layout: res[(i-start)*groupSize + lp] for each i in range and lp in [0, groupSize).
void evalFullDomainRangeDCF(int party, GroupElement *res, const DCFKeyPack &key,
                            uint64_t start, uint64_t len, int splitDepth = -1);


// Compute sum of evaluation over a contiguous range [start, start+len).
// Output: sums[lp] = sum over range of (eval(i)[lp])
void evalFullDomainSumDCF(int party, GroupElement *sums, const DCFKeyPack &key,
                            uint64_t start, uint64_t len, int splitDepth = -1);

std::pair<DualDCFKeyPack, DualDCFKeyPack> keyGenDualDCF(int Bin, int Bout, int groupSize, GroupElement idx, GroupElement *payload1, GroupElement *payload2);

std::pair<DualDCFKeyPack, DualDCFKeyPack> keyGenDualDCF(int Bin, int Bout, GroupElement idx, GroupElement payload1, GroupElement payload2);

void evalDualDCF(int party, GroupElement* res, GroupElement idx, const DualDCFKeyPack &key);
