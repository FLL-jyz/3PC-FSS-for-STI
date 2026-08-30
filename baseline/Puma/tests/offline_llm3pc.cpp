#include "party3pc.h"
#include "rss_protocols.h"
#include "test_base.h"
#include "llmConfig.h"
#include <vector>
#include <string>
#include "fss.h"
#include "2pcdpf.h"
#include <algorithm>
#include <omp.h>


static uint32_t batch_size = 1;
static bool print_res = false;
static int loop = print_res ? 1 : 5;
static int warmup_rounds = print_res ? 0 : 3;
static constexpr size_t BLOCK_BYTES = 16;
static bool Verification = false;

static size_t bytes_for_bits(int bits)
{
    return (bits + 7) / 8;
}

// Calculate DPF key size (in bytes)
// Parameters:
//  Bin: input domain bits
//  Bout: output/bucket bits
//  groupSize: number of groups (for g array length)
// For DPF, per user's instruction we count these fields as key material:
//  block *k0, block *k1   -> each length Bin+1, each block is 16 bytes
//  GroupElement *g         -> length groupSize, element size = bytes_for_bits(Bout)
//  std::array<std::array<block,2>,2> cs -> 4 blocks

static inline size_t dpf_key_size_bytes(int Bin, int Bout, int groupSize, bool Verification = false)
{
    size_t blocks_k = Bin + 1; // k0  send to 3 parties
    size_t bytes = blocks_k * BLOCK_BYTES;
    // g array: groupSize elements of Bout bits -> count bytes
    bytes += groupSize * bytes_for_bits(Bout);
    // // r
    // bytes += groupSize * bytes_for_bits(Bin);
    if (Verification)
    {
        // Bin * g and alpha check
        bytes += 2 * groupSize * bytes_for_bits(Bin);
        // cs: 2 blocks
        bytes += 2 * BLOCK_BYTES;
    }
    return bytes;
}

// Calculate DCF key size (in bytes)
// Fields to count (per user's instruction):
//  block *k0, *k1 -> each length Bin+1
//  GroupElement *g -> length groupSize
//  GroupElement *v -> length Bin * groupSize,Vcw
//  cs: 4 blocks

static inline size_t dcf_key_size_bytes(int Bin, int Bout, int groupSize, bool Verification = false)
{
    size_t blocks_k = Bin * 2 + 1; // k0 and k1
    size_t bytes = blocks_k * BLOCK_BYTES;
    // g
    bytes += groupSize * bytes_for_bits(Bout);
    // v
    bytes += groupSize * (Bin - 1) * bytes_for_bits(Bin);
    // r
    bytes += 2 * groupSize * bytes_for_bits(Bin);
    if (Verification)
    {
        // Bin * g and alpha check
        bytes += 2 * groupSize * bytes_for_bits(Bin);
        // cs: 2 blocks
        bytes += 2 * BLOCK_BYTES;
    }
    return bytes;
}

static inline bool add_overflow_size_t(size_t a, size_t b, size_t *out)
{
    if (out == nullptr)
    {
        return true;
    }
    if (b > std::numeric_limits<size_t>::max() - a)
    {
        return true;
    }
    *out = a + b;
    return false;
}

static inline bool mul_overflow_size_t(size_t a, size_t b, size_t *out)
{
    if (out == nullptr)
    {
        return true;
    }
    if (a == 0 || b == 0)
    {
        *out = 0;
        return false;
    }
    if (a > std::numeric_limits<size_t>::max() / b)
    {
        return true;
    }
    *out = a * b;
    return false;
}

// bit_len = cmp_size,count = number of comparisons
static TwoPcDPFKeyPack offline_pointcmp(int bit_len, size_t count, bool Verification = false, size_t extra_comm = 0)
{
    auto &party = Party3PC::getInstance();
    // Use DPF key size (Bin = Bout = bit_len, groupSize = 1)
    size_t key_size = dpf_key_size_bytes(bit_len, bit_len, 1, Verification);
    size_t total_key_bytes_mul = 0;
    size_t total_key_bytes = 0;
    if (mul_overflow_size_t(count, key_size, &total_key_bytes_mul) ||
        add_overflow_size_t(total_key_bytes_mul, extra_comm, &total_key_bytes))
    {
        throw std::overflow_error("offline_pointcmp: total_key_bytes overflow");
    }

    // Generate DPF keys `count` times to simulate offline generation cost.
    // We interleave generation and communication to prevent connection timeouts
    // when count is very large (e.g. millions).
    const size_t COMM_CHUNK_SIZE = 1 << 20; // 1MB logic chunks
    // Calculate how many keys fit into roughly one CHUNK.
    // Ensure we process at least one batch of KEY_BATCH size.
    size_t keys_per_comm_chunk = 1;
    if (key_size > 0) {
        keys_per_comm_chunk = COMM_CHUNK_SIZE / key_size;
    }
    if (keys_per_comm_chunk == 0) keys_per_comm_chunk = 1;

    // Align to 256 for friendly OpenMP batching
    keys_per_comm_chunk = ((keys_per_comm_chunk + 255) / 256) * 256;
    
    // Buffer for communication simulation
    // We use total_key_bytes to size it initially if small, else clamp to CHUNK
    size_t buffer_size_alloc = std::min((size_t)COMM_CHUNK_SIZE, total_key_bytes);
    if (buffer_size_alloc == 0) buffer_size_alloc = 1024; // safety
    std::vector<char> buf(buffer_size_alloc, 0);

    size_t generated = 0;
    size_t accumulated_sim_bytes = 0;
    
    // Helper to simulate sending 'bytes' amount of data (sending twice as per original logic)
    auto simulate_send_recv = [&](size_t bytes_to_send) {
        for(int round = 0; round < 2; ++round) {
            size_t rem = bytes_to_send;
            while (rem)
            {
                size_t t = std::min(rem, buf.size());
                Tensor<char> send_t(buf.data(), {(uint32_t)t});
                Tensor<char> recv_t({(uint32_t)t});
                std::thread t1([&]() { 
                    try { party.send_tensor_to<char>(party.next_party_id, send_t); } 
                    catch (std::exception &e) { std::cerr << "send_tensor_to error: " << e.what() << std::endl; } 
                });
                try {
                    party.recv_tensor_from<char>(party.pre_party_id, recv_t);
                } catch (std::exception &e) {
                    std::cerr << "recv_tensor_from error: " << e.what() << "; expected " << t << std::endl;
                }
                t1.join();
                rem -= t;
            }
        }
    };

    while (generated < count)
    {
        size_t this_batch = std::min(keys_per_comm_chunk, count - generated);
        
        // 1. Compute keys (CPU simulation)
        #pragma omp parallel for
        for (long long ii = 0; ii < (long long)this_batch; ++ii)
        {
            size_t i = generated + (size_t)ii;
            GroupElement idx(0, bit_len);
            GroupElement payload(0, bit_len);
            auto keys = keyGenTwoPcDPF(bit_len, bit_len, idx, payload, GenParams(Verification));
            (void)i;
        }
        
        // 2. Communicate (Network simulation)
        // Calculate bytes for this batch
        size_t current_bytes = this_batch * key_size;
        
        // Check if we are at the end and need to cover the remainder of total_key_bytes
        // (which includes extra_comm and any alignment differences)
        if (generated + this_batch == count) {
             size_t remaining_total = total_key_bytes - accumulated_sim_bytes;
             // Use the remaining bytes if larger than current calc (handles extra_comm)
             if (remaining_total > current_bytes) {
                 current_bytes = remaining_total; 
             }
        }
        
        if (current_bytes > 0) {
            simulate_send_recv(current_bytes);
            accumulated_sim_bytes += current_bytes;
        }

        generated += this_batch;
    }
    
    // If for some reason we finished generation but didn't match total_key_bytes 
    // (e.g. if key_size calc was slightly off or extra_comm passed incorrectly),
    // handle remainder here.
    if (accumulated_sim_bytes < total_key_bytes) {
        simulate_send_recv(total_key_bytes - accumulated_sim_bytes);
    }
    // Generate one keypair to return a usable DPF key for evaluation.
    auto single_keys = keyGenTwoPcDPF(bit_len, bit_len, GroupElement(0, bit_len), GroupElement(0, bit_len), GenParams(Verification));
    // Return the first share as the representative key for eval calls.
    return std::move(single_keys.first);

    // if (Verification){
    //     DCFKeyPack kp0 = DCFKeyPack();
    //     GroupElement res(0, bit_len);
    //     GroupElement idx(0, bit_len);
    //     size_t beta = 0;
    //     size_t alaha_check = 0;
    //     // 1.Full-domain evaluation to simulate verification cost
    //     #pragma omp parallel for
    //     for (size_t i = 0; i < count; ++i) {
    //         GroupElement idx(0, bit_len);
    //         for(size_t j = 0; j < (1 << bit_len); ++j){
    //             idx = GroupElement(j, bit_len);
    //             evalDCF(party.party_id, &res, idx, kp0,Verification);
    //         }
    //         beta += res.value;
    //         alaha_check += res.value * i;
    //     }

    //     // 2. Simulate hash verification cost and Open t, s
    //     size_t hash_len = (1 << bit_len) * 2 * BLOCK_BYTES;
    //     {
    //         size_t t = hash_len + 2 * bytes_for_bits(bit_len);
    //         Tensor<char> send_t({(uint32_t)t});
    //         Tensor<char> recv_t({(uint32_t)t});
    //         std::thread st([&](){ party.send_tensor_to<char>(party.next_party_id, send_t); });
    //         party.recv_tensor_from<char>(party.pre_party_id, recv_t);
    //         st.join();
    //     }

    // }
}

// bit_len = table_size
static void offline_lut(int bit_len, size_t count, bool Verification = false)
{
    auto &party = Party3PC::getInstance();
    // 1. VDPF Keys
    TwoPcDPFKeyPack kp0 = offline_pointcmp(bit_len, 3*count, Verification);
    if (not Verification)
    {
        GroupElement res(0, bit_len);
        GroupElement idx(0, bit_len);
#pragma omp parallel for
        for (size_t i = 0; i < 3*count; ++i)
        {
            GroupElement idx(0, bit_len);
            for (size_t j = 0; j < (1 << bit_len); ++j)
            {
                idx = GroupElement(j, bit_len);
                evalTwoPcDPF(party.party_id, &res, idx, kp0);
            }
        }
        // free dynamically allocated buffers inside kp0
        freeTwoPcDPFKeyPack(kp0);
    }
}

static void offline_ucmp(int bit_len, size_t count, bool Verification = false, size_t extra_comm = 0)
{
    // 1. VDPF Keys
    auto &party = Party3PC::getInstance();
    // Ensure key_len is large enough to cover the range [0, 2*bit_len)
    int key_len = static_cast<int>(std::floor(std::log2(2 * bit_len))) + 1;
    
    // Generate and retrieve a usable DPF key for evaluation (instead of null pointers).
    TwoPcDPFKeyPack kp0 = offline_pointcmp(key_len, count * bit_len, Verification, extra_comm);
    // 2. eval
    #pragma omp parallel for
    for (size_t i = 0; i < 2 * bit_len; ++i)
    {
        
        GroupElement idx(0, key_len);
        for (size_t j = 0; j < count * bit_len; ++j)
        {
            idx = GroupElement(i, key_len);
            GroupElement res(0, key_len);
            // Each party performs evalVDPF to simulate eval cost
            evalTwoPcDPF(party.party_id, &res, idx, kp0);
        }
    }
    // free dynamically allocated buffers inside kp0
    freeTwoPcDPFKeyPack(kp0);
}

static void offline_drelu(int bit_len, size_t count, bool Verification = false)
{
    auto &party = Party3PC::getInstance();
    // 1. Shares of c,
    size_t share_bytes = count * bytes_for_bits(bit_len) * 2;
    offline_ucmp(bit_len, 3 * count, Verification, share_bytes); //should count * 3
    

    // 2. Reveal delta_c, delta_y
    // if (Verification){
    //     size_t a_val=0, b_val=0;
    //     RSSVectorMyType a(count);
    //     RSSVectorSmallType theta(count);
    //     // Initialize a with some shares
    //     #pragma omp parallel for
    //     for(long long ii=0; ii<(long long)count; ++ii) {
    //         size_t i = (size_t)ii;
    //         a[i].first = a_val;
    //         a[i].second = b_val;
    //     }
    //     funcWrap(a, theta, count);
    //     // if (party.party_id == 0) std::cout << "funcWrap executed successfully." << std::endl;

    //     // verify delta_c, delta_y according to protocol in the image:
    //     RSSTensor<size_t> delta_c_shared({(uint32_t)count});
    //     RSSTensor<size_t> delta_y_shared({(uint32_t)count});
    //     #pragma omp parallel for
    //     for (long long ii = 0; ii < (long long)count; ++ii) {
    //         size_t i = (size_t)ii;
    //         delta_c_shared.first.data[i] = 1 - theta[i].first;
    //         delta_c_shared.second.data[i] = 1 - theta[i].second;
    //         delta_y_shared.first.data[i] = 1 - theta[i].first;
    //         delta_y_shared.second.data[i] = 1 - theta[i].second;
    //     }
    //     __int128 total_key_bytes_128 = ( __int128)2 * ( __int128)count * ( __int128)bytes_for_bits(bit_len);
    //     if (total_key_bytes_128 > ( __int128)std::numeric_limits<size_t>::max()){
    //     } else {
    //         size_t total_key_bytes = (size_t)total_key_bytes_128;
    //         const size_t CHUNK = (size_t)1 << 20; // 1MB
    //         std::vector<char> buf(std::min(CHUNK, total_key_bytes), 0);
    //         size_t rem = total_key_bytes;
    //         while (rem) {
    //             size_t t = std::min(rem, buf.size());
    //             Tensor<char> send_t({(uint32_t)t});
    //             std::memcpy(send_t.data, buf.data(), t);
    //             Tensor<char> recv_t({(uint32_t)t});
    //             std::thread st([&](){ party.send_tensor_to<char>(party.next_party_id, send_t); });
    //             party.recv_tensor_from<char>(party.pre_party_id, recv_t);
    //             st.join();
    //             rem -= t;
    //         }
    //         rem = total_key_bytes;
    //         while (rem) {
    //             size_t t = std::min(rem, buf.size());
    //             Tensor<char> send_t({(uint32_t)t});
    //             std::memcpy(send_t.data, buf.data(), t);
    //             Tensor<char> recv_t({(uint32_t)t});
    //             std::thread st([&](){ party.send_tensor_to<char>(party.next_party_id, send_t); });
    //             party.recv_tensor_from<char>(party.pre_party_id, recv_t);
    //             st.join();
    //             rem -= t;
    //         }
    //     }

    // }
}

static void run_offline_softmax(int party_id,size_t exp_num, size_t max_num){
    omp_set_num_threads(16);
   // Softmax： uses UCMP for Max
    offline_drelu(31, max_num);
    // Assuming Softmax uses UCMP for e^x
    offline_drelu(11, exp_num);
    offline_lut(4, exp_num); // compute >>z
}

static void run_offline_Inv(int party_id,size_t inv_num){
    omp_set_num_threads(16);
// Inv: One for base LUT; One for Inv LUT
    offline_drelu(31, 2 * inv_num);
    offline_lut(2, inv_num);
    offline_lut(12, inv_num);
}

static void run_offline_MHA(int party_id,size_t exp_num, size_t inv_num, size_t max_num){
   // Softmax： uses UCMP for Max
    omp_set_num_threads(16);
    offline_drelu(31, max_num);
    // Inv: One for base LUT; One for Inv LUT
    offline_drelu(31, 2 * inv_num);
    offline_lut(2, inv_num);
    offline_lut(12, inv_num);
    // Assuming Softmax uses UCMP for e^x
    offline_drelu(11, exp_num);
    offline_lut(4, exp_num); // compute >>z

}

static void run_offline_ln(int party_id,size_t ln_num){
   // Rsqt:One for base LUT; One for rsqt LUT; One for exsqrt
   omp_set_num_threads(16);
    offline_drelu(31, 2 * ln_num);
    offline_lut(2, ln_num);
    offline_lut(2, ln_num);
    offline_lut(12, ln_num);
}

static void run_offline_ffn(int party_id,size_t gelu_num){
   //Gelu
    omp_set_num_threads(16);
    offline_drelu(15, gelu_num * 3);
    offline_lut(8, gelu_num);
}

static void run_offline_lut_12(int party_id,size_t num){
    omp_set_num_threads(16);
    offline_lut(12, num);
}


static void offline_tf_test(int party_id){
    Party3PC::getInstance().load_param<uint64_t>();
    TransformerConfig tf_config("transformer", 64);
    omp_set_num_threads(16);
    size_t seq_len = tf_config.seq_len;
    size_t hidden = tf_config.hidden_size;
    size_t intermediate = tf_config.intermediate_size;
    size_t heads = tf_config.num_attention_heads;
    // GELU (16-bit LUT)
    // FFN structure: hidden -> intermediate -> (GELU) -> hidden
    size_t gelu_count =  seq_len * intermediate;

    // Softmax (32-bit LUT)
    // Attention scores matrix: batch * heads * seq_len * seq_len
    size_t softmax_exp_count =  heads * seq_len * seq_len;
    size_t softmax_inv_count =  heads * seq_len;

    // LayerNorm (32-bit LUT)
    // Encoder layers have 2 LNs each.
    // bert_test in llm3pc.cpp adds 1 extra SecMixedLayerNorm.
    size_t ln_count =  seq_len;

    // Softmax Max (UCMP)
    // Finding max of seq_len elements requires seq_len - 1 comparisons.
    size_t softmax_max_count =  heads * seq_len * (seq_len - 1);

    // for (uint32_t num = 10000; num <= 10000; num *= 10)
    // {
    //     if (party_id == 0)
    //     {
    //         std::cout << "********************** num = " << num << " **********************" << std::endl;
    //     }
    //     test_func<Party3PC>(run_offline_softmax, "Offline Softmax", loop, warmup_rounds, party_id, num, num-1);
    //     Party3PC::getInstance().sync();
    //     test_func<Party3PC>(run_offline_Inv, "Offline Inv", loop, warmup_rounds, party_id, num);
    //     Party3PC::getInstance().sync();
    //     test_func<Party3PC>(run_offline_ffn, "Offline GELU", loop, warmup_rounds, party_id, num);
    //     Party3PC::getInstance().sync();
    //     test_func<Party3PC>(run_offline_ln, "Offline Reciprocal Square Root", loop, warmup_rounds, party_id, num);
    //     Party3PC::getInstance().sync();
    //     test_func<Party3PC>(run_offline_lut_12, "Offline LUT-12", loop, warmup_rounds, party_id, num);
    //     Party3PC::getInstance().sync();
        
    // }
    test_func<Party3PC>(run_offline_MHA, "Offline MHA", loop, warmup_rounds, party_id, softmax_exp_count/2, softmax_inv_count/2, softmax_max_count/2);
    Party3PC::getInstance().sync();
    test_func<Party3PC>(run_offline_MHA, "Offline MMHA", loop, warmup_rounds, party_id, softmax_exp_count/2, softmax_inv_count/2, softmax_max_count/2/2);
    Party3PC::getInstance().sync();
    test_func<Party3PC>(run_offline_ffn, "Offline FFN", loop, warmup_rounds, party_id, gelu_count/2);
    Party3PC::getInstance().sync();
    test_func<Party3PC>(run_offline_ln, "Offline LayerNorm", loop, warmup_rounds, party_id, ln_count);
    Party3PC::getInstance().sync();
}

static void run_offline_bert(int party_id) {
    // set OpenMP threads for this workload
    omp_set_num_threads(16);
    TransformerConfig config("bert-base", 128);
    size_t layers = config.num_layers;
    size_t seq_len = config.seq_len;
    size_t hidden = config.hidden_size;
    size_t intermediate = config.intermediate_size;
    size_t heads = config.num_attention_heads;
    size_t batch = batch_size;

    // GELU (16-bit LUT)
    // FFN structure: hidden -> intermediate -> (GELU) -> hidden
    size_t gelu_count = layers * batch * seq_len * intermediate;
    
    // Softmax (32-bit LUT)
    // Attention scores matrix: batch * heads * seq_len * seq_len
    size_t softmax_exp_count = layers * batch * heads * seq_len * seq_len;
    size_t softmax_inv_count = layers * batch * heads * seq_len;
    
    // LayerNorm (32-bit LUT)
    // Encoder layers have 2 LNs each.
    // bert_test in llm3pc.cpp adds 1 extra SecMixedLayerNorm.
    size_t ln_count = (layers * 2 + 1) * batch * seq_len;

    // Softmax Max (UCMP)
    // Finding max of seq_len elements requires seq_len - 1 comparisons.
    size_t softmax_max_count = layers * batch * heads * seq_len * (seq_len - 1);

    // cout << "Gelun number" << gelu_count << endl;   //4718592  3072*128*12
    // cout << "Softmax exp number" << softmax_exp_count << endl; //2359296    1536*128*12
    // cout << "Softmax inv number" << softmax_inv_count << endl; //18432  128*12*12
    // cout << "Ln number" << ln_count << endl;    //3200  128*25
    // cout << "Softmax max number" << softmax_max_count << endl;  //2340864   1524*128*12
    
    //Gelu
    offline_drelu(15, 36864 * 3/4);
    offline_lut(8, 36864/4);

    // Rsqt:One for base LUT; One for rsqt LUT; One for exsqrt
    offline_drelu(31, 2 * 25/4);
    offline_lut(2, 25/4); //ex_base
    offline_lut(2, 25/4); //exsqrt_base
    offline_lut(12, 25/4);

    //Inv: One for base LUT; One for Inv LUT
    offline_drelu(31, 2 * 12*12/4);
    offline_lut(2, 12*12/4); //ex_base
    offline_lut(12, 12*12/4); 

    // Softmax: uses UCMP for Max
    offline_drelu(31, 1524*12/4);
    // Assuming Softmax uses UCMP for e^x
    offline_drelu(11, 1536*12/4);
    offline_lut(4, 1536*12/4); // compute >>z
}

static void run_offline_gpt2(int party_id) {
    // set OpenMP threads for this workload
    omp_set_num_threads(16);
    TransformerConfig config("gpt2", 64);
    size_t layers = config.num_layers;
    size_t seq_len = config.seq_len;
    size_t hidden = config.hidden_size;
    size_t intermediate = config.intermediate_size;
    size_t heads = config.num_attention_heads;
    size_t batch = batch_size;
	

    // GELU (16-bit LUT)
    size_t gelu_count = layers * batch * seq_len * intermediate;
    
    // Softmax (32-bit LUT)
    // Attention scores matrix: batch * heads * seq_len * seq_len
    size_t softmax_exp_count = layers * batch * heads * seq_len * seq_len;
    size_t softmax_inv_count = layers * batch * heads * seq_len;
    
    // LayerNorm (32-bit LUT)
    // GPT2 has 2 LNs per layer + 1 final LN (ln_f).
    size_t ln_count = (layers * 2 + 1) * batch * seq_len;

    // Softmax Max (UCMP)
    size_t softmax_max_count = layers * batch * heads * seq_len * (seq_len - 1);
    
    // cout << "Gelun number" << gelu_count << endl;   //2359296  3072*64*12
    // cout << "Softmax exp number" << softmax_exp_count << endl; //589824    768*64*12
    // cout << "Softmax inv number" << softmax_inv_count << endl; //9216  64*12*12
    // cout << "Ln number" << ln_count << endl;    //1600  64*25
    // cout << "Softmax max number" << softmax_max_count << endl;  //580608   756*64*12

    //Gelu
    offline_drelu(15, 3072 * 12 * 3/8);
    offline_lut(8, 3072 * 12/8);

    // Rsqt:One for base LUT; One for rsqt LUT; One for exsqrt
    offline_drelu(31, 2 * 25/8);
    offline_lut(2, 25/8);
    offline_lut(2, 25/8);
    offline_lut(12, 25/8);

    // Assuming Softmax uses UCMP for Max
    offline_drelu(31, 756 * 12/8);
    // Inv: One for base LUT; One for Inv LUT
    offline_drelu(31, 2 * 12 * 12/8);
    offline_lut(2, 12 * 12/8);
    offline_lut(12, 12 * 12/8);

    // Assuming Softmax uses UCMP for e^x
    offline_drelu(11, 768 * 12/8);
    offline_lut(4, 768 * 12/8);
}

void OfflineLLM3pcTest::main_test(int party_id)
{
    fss_init();
    offline_tf_test(party_id);
    Party3PC::getInstance().sync();
    // test_func<Party3PC>(run_offline_bert, "Offline BERT", loop, warmup_rounds, party_id);
    // Party3PC::getInstance().sync();
    // test_func<Party3PC>(run_offline_gpt2, "Offline GPT-2", loop, warmup_rounds, party_id);
}
