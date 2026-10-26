#include "groto_dpf.h"

#include <array>
#include <stdexcept>
#include <vector>

#include "2pcdpf.h"

namespace groto_dpf
{

namespace
{

constexpr uint64_t kLeafParentArity = 4;

inline block descendCorrectedSeed(const block &s, const block &cw, u8 keep)
{
    static const block notThreeBlock = toBlock(~0, ~3);
    static const block TwoBlock = toBlock(0, 2);
    static const block ThreeBlock = toBlock(0, 3);
    static const block blocks[4] = {ZeroBlock, TwoBlock, OneBlock, ThreeBlock};

    const u8 tPrevious = lsb(s);
    const block scw = (cw & notThreeBlock);
    block ds[] = {((cw >> 1) & OneBlock), (cw & OneBlock)};
    const block mask = zeroAndAllOne[tPrevious];
    const block ss = s & notThreeBlock;

    AES ak(ss);
    block ct[2];
    ak.ecbEncTwoBlocks(blocks + 2 * keep, ct);

    return ((scw ^ ds[keep]) & mask) ^ ct[0];
}

inline void seedToShares(const block &s, int Bout, int groupSize, std::vector<GroupElement> *out)
{
    static const block notThreeBlock = toBlock(~0, ~3);
    std::vector<uint64_t> converted(groupSize, 0ULL);
    convert(Bout, groupSize, s & notThreeBlock, converted.data());

    out->assign(groupSize, GroupElement(0, Bout));
    for (int i = 0; i < groupSize; ++i)
    {
        (*out)[i] = GroupElement(converted[i], Bout);
    }
}

inline void xorAccumulate(int groupSize, GroupElement *acc, const std::vector<GroupElement> &delta)
{
    for (int i = 0; i < groupSize; ++i)
    {
        acc[i].value = acc[i].value ^ delta[i].value;
        mod(acc[i]);
    }
}

inline void validateBooleanPayload(int Bout, int groupSize, const GroupElement *payload)
{
    if (payload == nullptr)
    {
        throw std::invalid_argument("payload must not be null");
    }

    for (int i = 0; i < groupSize; ++i)
    {
        if (payload[i].bitsize != Bout || payload[i].value != 1ULL)
        {
            throw std::invalid_argument("groto_dpf currently supports only boolean payload=1");
        }
    }
}

inline void geqToLtShares(int party, int Bout, int groupSize, const GroupElement *geqShares, GroupElement *ltShares)
{
    const GroupElement one(1, Bout);
    const GroupElement zero(0, Bout);
    for (int i = 0; i < groupSize; ++i)
    {
        // If geqShares reconstructs to 1[x >= alpha], then:
        // 1[x < alpha] = 1 - 1[x >= alpha].
        if (party == 0)
        {
            ltShares[i] = one - geqShares[i];
        }
        else
        {
            ltShares[i] = zero - geqShares[i];
        }
    }
}

} // namespace

std::pair<GrotoDPFKeyPack, GrotoDPFKeyPack> keyGenGrotoDPF(
    int Bin,
    int Bout,
    int groupSize,
    GroupElement idx,
    GroupElement *payload,
    const GenParams &gen)
{
    validateBooleanPayload(Bout, groupSize, payload);
    return keyGenTwoPcDPF(Bin, Bout, groupSize, idx, payload, gen);
}

std::pair<GrotoDPFKeyPack, GrotoDPFKeyPack> keyGenGrotoDPF(
    int Bin,
    int Bout,
    GroupElement idx,
    GroupElement payload,
    const GenParams &gen)
{
    if (payload.bitsize != Bout || payload.value != 1ULL)
    {
        throw std::invalid_argument("groto_dpf currently supports only boolean payload=1");
    }
    return keyGenTwoPcDPF(Bin, Bout, idx, payload, gen);
}

void evalGrotoDPF(
    int party,
    GroupElement *res,
    GroupElement idx,
    const GrotoDPFKeyPack &key,
    bool enableVerify)
{
    (void)enableVerify;
    if (key.Bin >= 63)
    {
        throw std::invalid_argument("groto_dpf eval currently supports Bin < 63");
    }

    const uint64_t domain = (1ULL << key.Bin);
    const uint64_t x = idx.value & (domain - 1);

    std::vector<GroupElement> geqShares(key.groupSize, GroupElement(0, key.Bout));
    uint64_t l = 0;
    uint64_t r = domain;
    int level = 0;
    int prevDirection = 0; // Convention: start with "left".
    block s = key.k[0];

    std::vector<GroupElement> nodeShares;
    while ((r - l) > kLeafParentArity && level < key.Bin)
    {
        const uint64_t mid = l + ((r - l) >> 1);
        const int curDirection = (x < mid) ? 0 : 1;

        const block nextS = descendCorrectedSeed(s, key.k[level + 1], static_cast<u8>(curDirection));

        // Core rule: use corrected seed on this step, then XOR if direction changes.
        if (curDirection != prevDirection)
        {
            seedToShares(nextS, key.Bout, key.groupSize, &nodeShares);
            xorAccumulate(key.groupSize, geqShares.data(), nodeShares);
        }

        s = nextS;
        if (curDirection == 0)
        {
            r = mid;
        }
        else
        {
            l = mid;
        }
        prevDirection = curDirection;
        ++level;
    }

    // Leaf remainder: XOR only corrected seeds of covered leaves in leaf substring.
    for (uint64_t t = l; t <= x && t < r; ++t)
    {
        block leafSeed = s;
        uint64_t ll = l;
        uint64_t rr = r;
        int leafLevel = level;
        while (leafLevel < key.Bin)
        {
            const uint64_t mid = ll + ((rr - ll) >> 1);
            const int dir = (t < mid) ? 0 : 1;
            leafSeed = descendCorrectedSeed(leafSeed, key.k[leafLevel + 1], static_cast<u8>(dir));
            if (dir == 0)
            {
                rr = mid;
            }
            else
            {
                ll = mid;
            }
            ++leafLevel;
        }

        seedToShares(leafSeed, key.Bout, key.groupSize, &nodeShares);
        xorAccumulate(key.groupSize, geqShares.data(), nodeShares);
    }

    geqToLtShares(party, key.Bout, key.groupSize, geqShares.data(), res);
}

void evalFullDomainGrotoDPF(
    int party,
    GroupElement *res,
    const GrotoDPFKeyPack &key)
{
    if (key.Bin >= 63)
    {
        throw std::invalid_argument("groto_dpf full-domain eval currently supports Bin < 63");
    }

    const uint64_t domain = (1ULL << key.Bin);
    for (uint64_t x = 0; x < domain; ++x)
    {
        GroupElement idx(x, key.Bin);
        evalGrotoDPF(party, res + x * key.groupSize, idx, key, false);
    }
}

GroupElement groto_dpf(
    int party,
    GroupElement idx,
    const GrotoDPFKeyPack &key,
    bool enableVerify)
{
    if (key.groupSize != 1)
    {
        throw std::invalid_argument("groto_dpf expects key.groupSize == 1");
    }

    GroupElement out[1];
    evalGrotoDPF(party, out, idx, key, enableVerify);
    return out[0];
}

} // namespace groto_dpf
