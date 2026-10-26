/*
CPU wrapper for Groto-style DPF/DCF API.
This module uses 2PC DPF as backend and computes DCF shares
through prefix-sum style evaluation.
*/

#pragma once

#include <utility>

#include "group_element.h"
#include "keypack.h"
#include "2pckeypack.h"

namespace groto_dpf
{

using GrotoDPFKeyPack = TwoPcDPFKeyPack;

std::pair<GrotoDPFKeyPack, GrotoDPFKeyPack> keyGenGrotoDPF(
    int Bin,
    int Bout,
    int groupSize,
    GroupElement idx,
    GroupElement *payload,
    const GenParams &gen = GenParams());

std::pair<GrotoDPFKeyPack, GrotoDPFKeyPack> keyGenGrotoDPF(
    int Bin,
    int Bout,
    GroupElement idx,
    GroupElement payload,
    const GenParams &gen = GenParams());

void evalGrotoDPF(
    int party,
    GroupElement *res,
    GroupElement idx,
    const GrotoDPFKeyPack &key,
    bool enableVerify = false);

void evalFullDomainGrotoDPF(
    int party,
    GroupElement *res,
    const GrotoDPFKeyPack &key);

// Single-entry helper for groupSize == 1 usage.
GroupElement groto_dpf(
    int party,
    GroupElement idx,
    const GrotoDPFKeyPack &key,
    bool enableVerify = false);

} // namespace groto_dpf
