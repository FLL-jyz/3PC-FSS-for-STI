#ifndef PARTY2PC_H
#define PARTY2PC_H

#include <cstring>
#include <any>
#include "../../common/network/include/comms.h"
#include "tensor.h"
// #include "macBuffer.h"
#include "loader.h"

class Party2PC
{
private:
    std::any loader; // Loder<T>
    bool is_malicious_ = false;

public:
    int party_id = -1;
    int next_party_id = -1;
    MyNetwork::Peer *peer_with_next = nullptr;
    // MacBuffer<T> mac_buffer;

    static Party2PC &getInstance();

    Party2PC() = default;
    Party2PC(const Party2PC &) = delete;
    Party2PC &operator=(const Party2PC &) = delete;

    void connect(int party_id, int server_port, const std::string &server_ip);

    void sync();

    void close();

    void semi_honest();

    void malicious();

    bool is_malicious();

    uint64_t bytes_sent();
    uint64_t bytes_received();

    uint64_t rounds_sent();
    uint64_t rounds_received();

    template <typename T>
    void send_to(int target_id, T *data, size_t size, size_t bit_width)
    {
        if (target_id == next_party_id)
        {
            this->peer_with_next->send_batched_input(data, size, bit_width);
        }
        else
        {
            throw std::invalid_argument("Invalid target_id");
        }
    }

    template <typename T>
    void recv_from(int source_id, T *data, size_t size, size_t bit_width)
    {
        if (source_id == next_party_id)
        {
            this->peer_with_next->recv_batched_input(data, size, bit_width);
        }
        else
        {
            throw std::invalid_argument("Invalid source_id");
        }
    }

    template <typename T>
    void send_value_to(int target_id, T value)
    {
        send_to(target_id, &value, 1, sizeof(T) * 8);
    }

    template <typename T>
    T recv_value_from(int source_id)
    {
        T value;
        recv_from(source_id, &value, 1, sizeof(T) * 8);
        return value;
    }

    template <typename T>
    void send_tensor_to(int target_id, Tensor<T> &tensor)
    {
        send_to(target_id, tensor.data, tensor.size(), sizeof(T) * 8);
    }

    template <typename T>
    void recv_tensor_from(int source_id, Tensor<T> &tensor)
    {
        recv_from(source_id, tensor.data, tensor.size(), sizeof(T) * 8);
    }

    // void mac_check();

    template <typename T>
    void load_param(const std::string &path)
    {
        Loader<T> loader(path);
        this->loader = loader;
    }

    template <typename T>
    void load_param()
    {
        Loader<T> loader;
        this->loader = loader;
    }

    template <typename T>
    Loader<T>& get_loader() {
        return std::any_cast<Loader<T>&>(loader);
    }
};

#endif
