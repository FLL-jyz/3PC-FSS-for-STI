#pragma once

#include <type_traits>
#include "tensor.h"
#include "../../../common/include/perf_stats.h"
#include "../../../common/network/include/comms.h"
#include "../../../secure_model/include/party2pc.h"

namespace ass_protocols
{
    struct A2BBooleanAdderBenchmarkResult
    {
        uint32_t sampled_values = 0;
        double seconds = 0.0;
        double values_per_sec = 0.0;
    };

    namespace detail
    {
        void a2b_boolean_adder_u16(uint16_t *data, uint32_t size, int party_id, const std::string &address, uint16_t port, uint32_t a2b_bitlength = 0);
        void a2b_boolean_adder_u32(uint32_t *data, uint32_t size, int party_id, const std::string &address, uint16_t port, uint32_t a2b_bitlength = 0);
        void a2b_boolean_adder_u64(uint64_t *data, uint32_t size, int party_id, const std::string &address, uint16_t port, uint32_t a2b_bitlength = 0);
    }

    A2BBooleanAdderBenchmarkResult benchmark_a2b_boolean_adder_u64(uint32_t size, uint32_t rounds = 1, uint32_t a2b_bitlength = 0);
    A2BBooleanAdderBenchmarkResult benchmark_a2b_boolean_adder_u64_cached(uint32_t size, uint32_t rounds = 1, uint32_t a2b_bitlength = 0);

    template <typename T>
    void a2b(Tensor<T> &x, MyNetwork::Peer &peer, int party_id, const char *perf_scope_prefix = nullptr, uint16_t a2b_port_offset = 1024, uint32_t a2b_bitlength = 0)
    {
        std::string perf_name_storage;
        const char *perf_name = "ass_protocols::a2b";
        if (perf_scope_prefix != nullptr)
        {
            perf_name_storage = std::string(perf_scope_prefix) + "::a2b";
            perf_name = perf_name_storage.c_str();
        }
        perf_stats::Scope perf_scope(perf_name);
        (void)peer;

        if (x.size() == 0)
        {
            return;
        }

        Party2PC &party = Party2PC::getInstance();
        always_assert(party.party_id == party_id);
        always_assert(party.server_port() >= 0);
        const uint16_t a2b_port = static_cast<uint16_t>(party.server_port() + a2b_port_offset);
        const uint32_t resolved_bitlength = a2b_bitlength == 0 ? static_cast<uint32_t>(sizeof(T) * 8) : a2b_bitlength;
        always_assert(resolved_bitlength > 0);
        always_assert(resolved_bitlength <= static_cast<uint32_t>(sizeof(T) * 8));

        if constexpr (std::is_same_v<T, uint16_t>)
        {
            detail::a2b_boolean_adder_u16(x.data, x.size(), party_id, party.server_ip(), a2b_port, resolved_bitlength);
        }
        else if constexpr (std::is_same_v<T, uint32_t>)
        {
            detail::a2b_boolean_adder_u32(x.data, x.size(), party_id, party.server_ip(), a2b_port, resolved_bitlength);
        }
        else if constexpr (std::is_same_v<T, uint64_t>)
        {
            detail::a2b_boolean_adder_u64(x.data, x.size(), party_id, party.server_ip(), a2b_port, resolved_bitlength);
        }
        else
        {
            static_assert(!std::is_same_v<T, T>, "Boolean-adder-backed a2b only supports uint16_t, uint32_t, and uint64_t");
        }
    }

    template <typename T>
    void b2a(Tensor<T> &x, MyNetwork::Peer &peer, int party_id, const char *perf_scope_prefix = nullptr)
    {
        std::string perf_name_storage;
        const char *perf_name = "ass_protocols::b2a";
        if (perf_scope_prefix != nullptr)
        {
            perf_name_storage = std::string(perf_scope_prefix) + "::b2a";
            perf_name = perf_name_storage.c_str();
        }
        perf_stats::Scope perf_scope(perf_name);
        uint32_t size = x.size();
        Tensor<T> r(x.shape);
        // Opening x + r is only safe when r is a fresh secret random mask.
        // The current placeholder keeps r = 0, so this path is functionally correct only if revealing the bit is acceptable.
        r.zero();

        Tensor<uint8_t> x_shift(x.shape), tmp(x.shape);
        for (int i = 0; i < size; i++)
        {
            x_shift.data[i] = (uint8_t)((x.data[i] + r.data[i]) % 2);
        }
        peer.send_batched_input(x_shift.data, size, 1);
        peer.recv_batched_input(tmp.data, size, 1);

        if (party_id == 0)
        {
            for (int i = 0; i < size; i++)
            {
                x.data[i] = r.data[i] - r.data[i] * (x_shift.data[i] ^ tmp.data[i]) * 2;
            }
        }
        else
        {
            T x_shift_restore;
            for (int i = 0; i < size; i++)
            {
                x_shift_restore = x_shift.data[i] ^ tmp.data[i];
                x.data[i] = x_shift_restore + r.data[i] - r.data[i] * x_shift_restore * 2;
            }
        }
    }

    /**
     * r > x
     * @tparam T
     * @param r_bits shared value of bits of r
     * @param x_shift plaintext value
     * @param res
     * @param peer
     * @param party_id
     */
    
}
