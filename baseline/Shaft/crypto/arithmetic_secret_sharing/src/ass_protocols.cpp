#include "ass_protocols.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <atomic>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <random>
#include <exception>

namespace ass_protocols
{
	namespace detail
	{
		namespace
		{
			struct BoolTripleShare
			{
				uint8_t a = 0;
				uint8_t b = 0;
				uint8_t c = 0;
			};

			template <typename T>
			inline uint32_t resolve_a2b_bitlength(uint32_t a2b_bitlength)
			{
				const uint32_t max_bitlength = static_cast<uint32_t>(sizeof(T) * 8);
				const uint32_t resolved = a2b_bitlength == 0 ? max_bitlength : a2b_bitlength;
				always_assert(resolved > 0);
				always_assert(resolved <= max_bitlength);
				return resolved;
			}

			template <typename T>
			inline T bitlength_mask(uint32_t bitlength)
			{
				const uint32_t max_bitlength = static_cast<uint32_t>(sizeof(T) * 8);
				if (bitlength >= max_bitlength)
				{
					return static_cast<T>(~static_cast<T>(0));
				}
				return static_cast<T>((static_cast<uint64_t>(1) << bitlength) - 1ULL);
			}

			template <typename T>
			inline void sanitize_input_bits(T *data, uint32_t size, uint32_t bitlength)
			{
				if (bitlength >= static_cast<uint32_t>(sizeof(T) * 8))
				{
					return;
				}
				const T mask = bitlength_mask<T>(bitlength);
				for (uint32_t i = 0; i < size; ++i)
				{
					data[i] &= mask;
				}
			}

			inline uint64_t mix_u64(uint64_t value)
			{
				value += 0x9e3779b97f4a7c15ULL;
				value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
				value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
				return value ^ (value >> 31);
			}

			template <typename T>
			inline std::string make_session_key(const std::string &address, uint16_t port, uint32_t bitlength)
			{
				return address + ":" + std::to_string(port) + ":" + std::to_string(sizeof(T) * 8) + ":" + std::to_string(bitlength);
			}

			inline uint64_t session_seed_for(const std::string &session_key)
			{
				return mix_u64(static_cast<uint64_t>(std::hash<std::string>{}(session_key)) ^ 0xA2B64A0B5EEDULL);
			}

			inline uint64_t boolean_and_gate_count(uint32_t size, uint32_t bitlength)
			{
				if (size == 0 || bitlength == 0)
				{
					return 0;
				}
				return static_cast<uint64_t>(size) * (2ULL * static_cast<uint64_t>(bitlength) - 1ULL);
			}

			inline uint64_t claim_gate_offset(const std::string &session_key, uint64_t gate_count)
			{
				static std::mutex *counter_mutex = new std::mutex();
				static std::map<std::string, uint64_t> *counters = new std::map<std::string, uint64_t>();
				std::lock_guard<std::mutex> lock(*counter_mutex);
				uint64_t &next_offset = (*counters)[session_key];
				const uint64_t base = next_offset;
				next_offset += gate_count;
				return base;
			}

			inline BoolTripleShare dealer_boolean_triple_share(uint64_t session_seed, uint64_t gate_index, int party_id)
			{
				uint64_t state = mix_u64(session_seed ^ mix_u64(gate_index + 1ULL));
				auto next_bit = [&state]() -> uint8_t {
					state = mix_u64(state + 0x9e3779b97f4a7c15ULL);
					return static_cast<uint8_t>(state & 1ULL);
				};

				const uint8_t u = next_bit();
				const uint8_t v = next_bit();
				const uint8_t w = static_cast<uint8_t>(u & v);

				const uint8_t u0 = next_bit();
				const uint8_t v0 = next_bit();
				const uint8_t w0 = next_bit();

				const uint8_t u1 = static_cast<uint8_t>(u ^ u0);
				const uint8_t v1 = static_cast<uint8_t>(v ^ v0);
				const uint8_t w1 = static_cast<uint8_t>(w ^ w0);

				if (party_id == 0)
				{
					return {u0, v0, w0};
				}
				return {u1, v1, w1};
			}

			inline uint8_t finalize_boolean_beaver_and(uint8_t opened_d, uint8_t opened_e,
											  const BoolTripleShare &triple,
											  int party_id)
			{
				uint8_t out = static_cast<uint8_t>(triple.c ^ (opened_e & triple.a) ^ (opened_d & triple.b));
				if (party_id == 1)
				{
					out ^= static_cast<uint8_t>(opened_d & opened_e);
				}
				return out;
			}

			inline void exchange_masks(MyNetwork::Peer &peer,
							  const std::vector<uint8_t> &local_masks,
							  std::vector<uint8_t> &remote_masks)
			{
				always_assert(!local_masks.empty());
				remote_masks.resize(local_masks.size());
				peer.send_batched_input(const_cast<uint8_t *>(local_masks.data()), static_cast<int>(local_masks.size()), 8);
				peer.recv_batched_input(remote_masks.data(), static_cast<int>(remote_masks.size()), 8);
			}

			template <typename T>
			void convert_with_boolean_adder(T *data,
								  uint32_t size,
								  int party_id,
								  MyNetwork::Peer &peer,
								  uint32_t bitlength,
								  const std::string &session_key)
			{
				if (size == 0)
				{
					return;
				}

				sanitize_input_bits(data, size, bitlength);
				const uint64_t session_seed = session_seed_for(session_key);
				uint64_t gate_cursor = claim_gate_offset(session_key, boolean_and_gate_count(size, bitlength));

				std::vector<T> packed_boolean_share(size, 0);
				std::vector<uint8_t> carry_share(size, 0);
				std::vector<uint8_t> next_carry_share(size, 0);
				std::vector<uint8_t> local_masks;
				std::vector<uint8_t> remote_masks;

				for (uint32_t bit = 0; bit < bitlength; ++bit)
				{
					const T bit_mask = static_cast<T>(static_cast<uint64_t>(1) << bit);

					if (bit == 0)
					{
						std::vector<BoolTripleShare> triples(size);
						local_masks.assign(static_cast<size_t>(size) * 2U, 0);

						for (uint32_t index = 0; index < size; ++index)
						{
							const uint8_t local_bit = static_cast<uint8_t>((data[index] >> bit) & 1U);
							if (local_bit != 0)
							{
								packed_boolean_share[index] |= bit_mask;
							}

							const BoolTripleShare triple = dealer_boolean_triple_share(session_seed, gate_cursor++, party_id);
							triples[index] = triple;

							const uint8_t a_share = party_id == 0 ? local_bit : 0;
							const uint8_t b_share = party_id == 1 ? local_bit : 0;
							local_masks[index] = static_cast<uint8_t>(a_share ^ triple.a);
							local_masks[size + index] = static_cast<uint8_t>(b_share ^ triple.b);
						}

						exchange_masks(peer, local_masks, remote_masks);

						for (uint32_t index = 0; index < size; ++index)
						{
							const uint8_t opened_d = static_cast<uint8_t>(local_masks[index] ^ remote_masks[index]);
							const uint8_t opened_e = static_cast<uint8_t>(local_masks[size + index] ^ remote_masks[size + index]);
							carry_share[index] = finalize_boolean_beaver_and(opened_d, opened_e, triples[index], party_id);
						}
						continue;
					}

					std::vector<BoolTripleShare> generate_triples(size);
					std::vector<BoolTripleShare> propagate_triples(size);
					local_masks.assign(static_cast<size_t>(size) * 4U, 0);

					for (uint32_t index = 0; index < size; ++index)
					{
						const uint8_t local_bit = static_cast<uint8_t>((data[index] >> bit) & 1U);
						const uint8_t sum_share = static_cast<uint8_t>(local_bit ^ carry_share[index]);
						if (sum_share != 0)
						{
							packed_boolean_share[index] |= bit_mask;
						}

						const BoolTripleShare generate_triple = dealer_boolean_triple_share(session_seed, gate_cursor++, party_id);
						const BoolTripleShare propagate_triple = dealer_boolean_triple_share(session_seed, gate_cursor++, party_id);
						generate_triples[index] = generate_triple;
						propagate_triples[index] = propagate_triple;

						const uint8_t a_share = party_id == 0 ? local_bit : 0;
						const uint8_t b_share = party_id == 1 ? local_bit : 0;
						local_masks[index] = static_cast<uint8_t>(a_share ^ generate_triple.a);
						local_masks[size + index] = static_cast<uint8_t>(b_share ^ generate_triple.b);
						local_masks[2U * size + index] = static_cast<uint8_t>(carry_share[index] ^ propagate_triple.a);
						local_masks[3U * size + index] = static_cast<uint8_t>(local_bit ^ propagate_triple.b);
					}

					exchange_masks(peer, local_masks, remote_masks);

					for (uint32_t index = 0; index < size; ++index)
					{
						const uint8_t generate_opened_d = static_cast<uint8_t>(local_masks[index] ^ remote_masks[index]);
						const uint8_t generate_opened_e = static_cast<uint8_t>(local_masks[size + index] ^ remote_masks[size + index]);
						const uint8_t propagate_opened_d = static_cast<uint8_t>(local_masks[2U * size + index] ^ remote_masks[2U * size + index]);
						const uint8_t propagate_opened_e = static_cast<uint8_t>(local_masks[3U * size + index] ^ remote_masks[3U * size + index]);

						const uint8_t generate_share = finalize_boolean_beaver_and(generate_opened_d, generate_opened_e, generate_triples[index], party_id);
						const uint8_t propagate_share = finalize_boolean_beaver_and(propagate_opened_d, propagate_opened_e, propagate_triples[index], party_id);
						next_carry_share[index] = static_cast<uint8_t>(generate_share ^ propagate_share);
					}

					carry_share.swap(next_carry_share);
				}

				for (uint32_t index = 0; index < size; ++index)
				{
					data[index] = packed_boolean_share[index];
				}
			}

			template <typename T>
			void run_uncached_boolean_adder_a2b_conversion(T *data, uint32_t size, int party_id, const std::string &address, uint16_t port, uint32_t a2b_bitlength)
			{
				const uint32_t resolved_bitlength = resolve_a2b_bitlength<T>(a2b_bitlength);
				const std::string session_key = make_session_key<T>(address, port, resolved_bitlength);
				std::unique_ptr<MyNetwork::Peer> peer;

				if (party_id == 0)
				{
					peer.reset(MyNetwork::waitForPeer(port));
				}
				else
				{
					peer.reset(new MyNetwork::Peer(address, port));
				}

				convert_with_boolean_adder(data, size, party_id, *peer, resolved_bitlength, session_key);
				peer->close();
			}

			template <typename T>
			class BooleanAdderA2BSession
			{
			public:
				static BooleanAdderA2BSession &get(const std::string &address, uint16_t port, int party_id, uint32_t a2b_bitlength)
				{
					static std::mutex *cache_mutex = new std::mutex();
					static std::map<std::string, std::unique_ptr<BooleanAdderA2BSession<T>>> *cache = new std::map<std::string, std::unique_ptr<BooleanAdderA2BSession<T>>>();
					const uint32_t resolved_bitlength = resolve_a2b_bitlength<T>(a2b_bitlength);
					const std::string cache_key = address + ":" + std::to_string(port) + ":" + std::to_string(party_id) + ":" + std::to_string(sizeof(T) * 8) + ":" + std::to_string(resolved_bitlength);

					std::lock_guard<std::mutex> lock(*cache_mutex);
					auto it = cache->find(cache_key);
					if (it == cache->end())
					{
						it = cache->emplace(cache_key, std::unique_ptr<BooleanAdderA2BSession<T>>(new BooleanAdderA2BSession<T>(address, port, party_id, resolved_bitlength))).first;
					}
					return *(it->second);
				}

				void convert(T *data, uint32_t size)
				{
					std::lock_guard<std::mutex> lock(run_mutex_);
					convert_with_boolean_adder(data, size, party_id_, *peer_, a2b_bitlength_, session_key_);
				}

			private:
				BooleanAdderA2BSession(const std::string &address, uint16_t port, int party_id, uint32_t a2b_bitlength)
					: party_id_(party_id),
					  a2b_bitlength_(a2b_bitlength),
					  session_key_(make_session_key<T>(address, port, a2b_bitlength))
				{
					if (party_id_ == 0)
					{
						peer_.reset(MyNetwork::waitForPeer(port));
					}
					else
					{
						peer_.reset(new MyNetwork::Peer(address, port));
					}
				}

				int party_id_;
				uint32_t a2b_bitlength_;
				std::string session_key_;
				std::unique_ptr<MyNetwork::Peer> peer_;
				std::mutex run_mutex_;
			};

			template <typename T>
			void verify_packed_boolean_result(const std::vector<T> &left_input,
									 const std::vector<T> &right_input,
									 const std::vector<T> &left_output,
									 const std::vector<T> &right_output,
									 uint32_t a2b_bitlength)
			{
				always_assert(left_input.size() == right_input.size());
				always_assert(left_input.size() == left_output.size());
				always_assert(left_input.size() == right_output.size());

				const uint32_t resolved_bitlength = resolve_a2b_bitlength<T>(a2b_bitlength);
				const T mask = bitlength_mask<T>(resolved_bitlength);
				for (size_t index = 0; index < left_input.size(); ++index)
				{
					const T expected = static_cast<T>((left_input[index] + right_input[index]) & mask);
					const T actual = static_cast<T>((left_output[index] ^ right_output[index]) & mask);
					always_assert(actual == expected);
				}
			}
		}

		void a2b_boolean_adder_u16(uint16_t *data, uint32_t size, int party_id, const std::string &address, uint16_t port, uint32_t a2b_bitlength)
		{
			BooleanAdderA2BSession<uint16_t>::get(address, port, party_id, a2b_bitlength).convert(data, size);
		}

		void a2b_boolean_adder_u32(uint32_t *data, uint32_t size, int party_id, const std::string &address, uint16_t port, uint32_t a2b_bitlength)
		{
			BooleanAdderA2BSession<uint32_t>::get(address, port, party_id, a2b_bitlength).convert(data, size);
		}

		void a2b_boolean_adder_u64(uint64_t *data, uint32_t size, int party_id, const std::string &address, uint16_t port, uint32_t a2b_bitlength)
		{
			BooleanAdderA2BSession<uint64_t>::get(address, port, party_id, a2b_bitlength).convert(data, size);
		}
	}

	namespace
	{
		A2BBooleanAdderBenchmarkResult benchmark_a2b_boolean_adder_u64_impl(uint32_t size, uint32_t rounds, uint32_t a2b_bitlength, bool warm_cache)
		{
			static std::atomic<uint16_t> benchmark_port_counter{42000};
			const uint32_t sample_size = std::max<uint32_t>(size, 1U);
			const uint32_t sample_rounds = std::max<uint32_t>(rounds, 1U);
			const std::string address = "127.0.0.1";
			const uint16_t port = benchmark_port_counter.fetch_add(1);

			std::vector<uint64_t> left_input(sample_size, 0);
			std::vector<uint64_t> right_input(sample_size, 0);
			std::mt19937_64 rng(0xA2B64A0BULL);
			for (uint32_t i = 0; i < sample_size; ++i)
			{
				left_input[i] = rng();
				right_input[i] = rng();
			}

			auto run_pair = [&](uint32_t benchmark_rounds) {
				std::exception_ptr left_error;
				std::exception_ptr right_error;
				std::vector<uint64_t> left_result(sample_size, 0);
				std::vector<uint64_t> right_result(sample_size, 0);

				std::thread server_thread([&]() {
					try
					{
						for (uint32_t round = 0; round < benchmark_rounds; ++round)
						{
							left_result = left_input;
							detail::a2b_boolean_adder_u64(left_result.data(), sample_size, 0, address, port, a2b_bitlength);
						}
					}
					catch (...)
					{
						left_error = std::current_exception();
					}
				});

				std::thread client_thread([&]() {
					try
					{
						for (uint32_t round = 0; round < benchmark_rounds; ++round)
						{
							right_result = right_input;
							detail::a2b_boolean_adder_u64(right_result.data(), sample_size, 1, address, port, a2b_bitlength);
						}
					}
					catch (...)
					{
						right_error = std::current_exception();
					}
				});

				server_thread.join();
				client_thread.join();

				if (left_error != nullptr)
				{
					std::rethrow_exception(left_error);
				}
				if (right_error != nullptr)
				{
					std::rethrow_exception(right_error);
				}

				detail::verify_packed_boolean_result(left_input, right_input, left_result, right_result, a2b_bitlength);
			};

			if (warm_cache)
			{
				run_pair(1);
			}

			const auto begin = std::chrono::steady_clock::now();
			run_pair(sample_rounds);
			const auto end = std::chrono::steady_clock::now();
			const double seconds = std::chrono::duration<double>(end - begin).count();

			A2BBooleanAdderBenchmarkResult result;
			result.sampled_values = sample_size;
			result.seconds = seconds / static_cast<double>(sample_rounds);
			result.values_per_sec = result.seconds > 0.0 ? static_cast<double>(sample_size) / result.seconds : 0.0;
			return result;
		}
	}

	A2BBooleanAdderBenchmarkResult benchmark_a2b_boolean_adder_u64(uint32_t size, uint32_t rounds, uint32_t a2b_bitlength)
	{
		static std::atomic<uint16_t> benchmark_port_counter{46000};
		const uint32_t sample_size = std::max<uint32_t>(size, 1U);
		const uint32_t sample_rounds = std::max<uint32_t>(rounds, 1U);
		const std::string address = "127.0.0.1";
		const uint16_t port = benchmark_port_counter.fetch_add(1);

		std::vector<uint64_t> left_input(sample_size, 0);
		std::vector<uint64_t> right_input(sample_size, 0);
		std::mt19937_64 rng(0xA2B64A0BULL);
		for (uint32_t i = 0; i < sample_size; ++i)
		{
			left_input[i] = rng();
			right_input[i] = rng();
		}

		std::exception_ptr left_error;
		std::exception_ptr right_error;
		std::vector<uint64_t> left_result(sample_size, 0);
		std::vector<uint64_t> right_result(sample_size, 0);
		const auto begin = std::chrono::steady_clock::now();

		std::thread server_thread([&]() {
			try
			{
				for (uint32_t round = 0; round < sample_rounds; ++round)
				{
					left_result = left_input;
					detail::run_uncached_boolean_adder_a2b_conversion<uint64_t>(left_result.data(), sample_size, 0, address, port, a2b_bitlength);
				}
			}
			catch (...)
			{
				left_error = std::current_exception();
			}
		});

		std::thread client_thread([&]() {
			try
			{
				for (uint32_t round = 0; round < sample_rounds; ++round)
				{
					right_result = right_input;
					detail::run_uncached_boolean_adder_a2b_conversion<uint64_t>(right_result.data(), sample_size, 1, address, port, a2b_bitlength);
				}
			}
			catch (...)
			{
				right_error = std::current_exception();
			}
		});

		server_thread.join();
		client_thread.join();

		if (left_error != nullptr)
		{
			std::rethrow_exception(left_error);
		}
		if (right_error != nullptr)
		{
			std::rethrow_exception(right_error);
		}

		detail::verify_packed_boolean_result(left_input, right_input, left_result, right_result, a2b_bitlength);

		const auto end = std::chrono::steady_clock::now();
		const double seconds = std::chrono::duration<double>(end - begin).count();

		A2BBooleanAdderBenchmarkResult result;
		result.sampled_values = sample_size;
		result.seconds = seconds / static_cast<double>(sample_rounds);
		result.values_per_sec = result.seconds > 0.0 ? static_cast<double>(sample_size) / result.seconds : 0.0;
		return result;
	}

	A2BBooleanAdderBenchmarkResult benchmark_a2b_boolean_adder_u64_cached(uint32_t size, uint32_t rounds, uint32_t a2b_bitlength)
	{
		return benchmark_a2b_boolean_adder_u64_impl(size, rounds, a2b_bitlength, true);
	}
}
