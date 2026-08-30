#include <iostream>
#include <cstring>
#include "party2pc.h"

Party2PC &Party2PC::getInstance()
{
    static Party2PC instance;
    return instance;
}

void Party2PC::connect(int party_id, int server_port, const std::string &server_ip)
{
    always_assert(party_id == 0 || party_id == 1);
    this->party_id = party_id;
    this->server_port_ = server_port;
    this->server_ip_ = server_ip;

    if (party_id == 0)
    {
        this->peer_with_next = MyNetwork::waitForPeer(server_port);
    }
    else
    {
        this->peer_with_next = new MyNetwork::Peer(server_ip, server_port);
    }

    this->next_party_id = 1 - party_id;
}

void Party2PC::sync()
{
    this->peer_with_next->sync();
}

void Party2PC::close()
{
    if (this->peer_with_next == nullptr)
    {
        return;
    }

    this->peer_with_next->close();
    delete this->peer_with_next;
    this->peer_with_next = nullptr;
}

void Party2PC::semi_honest()
{
    this->is_malicious_ = false;
}

void Party2PC::malicious()
{
    this->is_malicious_ = true;
}

bool Party2PC::is_malicious()
{
    return this->is_malicious_;
}

uint64_t Party2PC::bytes_sent()
{
    return this->peer_with_next->bytesSent();
}

uint64_t Party2PC::bytes_received()
{
    return this->peer_with_next->bytesReceived();
}

uint64_t Party2PC::rounds_sent()
{
    return this->peer_with_next->roundsSent();
}

uint64_t Party2PC::rounds_received()
{
    return this->peer_with_next->roundsReceived();
}

// template <typename T>
// void Party2PC<T>::mac_check()
// {
//     SecretShareTensor<T> value({static_cast<uint32_t>(mac_buffer.x.size())}), mac_value({
//                      static_cast<uint32_t>(mac_buffer.x.size())
//                  }), key(
//                      {static_cast<uint32_t>(mac_buffer.x.size())});

//     for (int i = 0; i < mac_buffer.x.size(); i++)
//     {
//         value.first.data[i] = mac_buffer.x[i].first;
//         value.second.data[i] = mac_buffer.x[i].second;
//         mac_value.first.data[i] = mac_buffer.mx[i].first;
//         mac_value.second.data[i] = mac_buffer.mx[i].second;
//         key.first.data[i] = mac_buffer.mac_key[i].first;
//         key.second.data[i] = mac_buffer.mac_key[i].second;
//     }

//     RSSProtocols::macCheck(value, mac_value, key, *this);

//     mac_buffer.x = std::vector<std::pair<T, T>>();
//     mac_buffer.mx = std::vector<std::pair<T, T>>();
//     mac_buffer.mac_key = std::vector<std::pair<T, T>>();
// }
