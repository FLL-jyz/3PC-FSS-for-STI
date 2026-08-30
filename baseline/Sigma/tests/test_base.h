#include <iostream>
#include "party2pc.h"
#include "perf_stats.h"
#include "tensor.h"
#include "twopc_protocols.h"
#include "globals.h"

class Test
{
public:
    virtual void main_test(int party_id) = 0;
    virtual void run(int party_id) = 0;
    virtual ~Test() = default;
};

class Test2PC : public Test
{
public:
    void run(int party_id) override
    {
        Party2PC &party = Party2PC::getInstance();
        if (party_id == 0)
        {
            std::cout << "P0" << std::endl;
            party.connect(0, kP0_Port, kP0_IP);
        }
        else if (party_id == 1)
        {
            std::cout << "P1" << std::endl;
            party.connect(1, kP0_Port, kP0_IP);
        }
        else
        {
            std::cerr << "Invalid party id for 2PC" << std::endl;
            return;
        }
        party.sync();

        main_test(party_id);

        party.close();
    }
};

template <typename P, typename F, typename... Args>
void test_func(F func, std::string test_name, int loop = 10, int warm_up = 5, Args &&...args)
{
    if (P::getInstance().party_id == 0)
    {
        std::cout << "====================Test for " << test_name << " ====================" << std::endl;
    }
    for (int i = 0; i < warm_up; i++)
    {
        func(std::forward<Args>(args)...);
    }

    uint64_t rounds_sent = P::getInstance().rounds_sent();
    uint64_t rounds_received = P::getInstance().rounds_received();
    uint64_t bytes_sent = P::getInstance().bytes_sent();
    uint64_t bytes_received = P::getInstance().bytes_received();

    struct timespec requestStart, requestEnd;
    clock_t start = clock();
    if (perf_stats::is_enabled())
    {
        perf_stats::reset();
    }
    clock_gettime(CLOCK_REALTIME, &requestStart);

    for (int i = 0; i < loop; i++)
    {
        func(std::forward<Args>(args)...);
    }

    clock_t clock_end = clock();
    clock_gettime(CLOCK_REALTIME, &requestEnd);

    if (P::getInstance().party_id == 0)
    {
        std::cout << "Time:" << std::endl;
        std::cout << "Wall Clock time: " << diff(requestStart, requestEnd) / loop << " sec\n";
        std::cout << "CPU time: " << (double)(clock_end - start) / CLOCKS_PER_SEC / loop << std::endl;

        std::cout << "Comm.: " << std::endl;
        std::cout << "\t Sent rounds:" << (P::getInstance().rounds_sent() - rounds_sent) / loop << "\t Sent Bytes:"
                  << convert_bytes_to_string((P::getInstance().bytes_sent() - bytes_sent) / loop) << std::endl;
        std::cout << "\t Recv rounds:" << (P::getInstance().rounds_received() - rounds_received) / loop << "\t Recv Bytes:"
                  << convert_bytes_to_string((P::getInstance().bytes_received() - bytes_received) / loop) << std::endl;
        if (perf_stats::is_enabled())
        {
            perf_stats::print_report(std::cout, static_cast<uint64_t>(loop));
        }
    }
}

template <typename T>
void open_print(twopc_protocols::ShareTensor<T> &x_share, int party_id, bool print_res)
{
    if (print_res)
    {
        Tensor<T> x_open(x_share.shape);

        twopc_protocols::restore(x_share, x_open);

        if (party_id == 0)
        {
            x_open.print();
        }
    }
}

class TwoPcProtocolTest : public Test2PC
{
    void main_test(int party_id) override;
};

class TwoPcProtocolBenchmarkTest : public Test2PC
{
    void main_test(int party_id) override;
};

class CNN2pcTest : public Test2PC
{
    void main_test(int party_id) override;
};

class LLM2pcTest : public Test2PC
{
    void main_test(int party_id) override;
};

class LLMAccTest : public Test2PC
{
    void main_test(int party_id) override;
};

class OfflineLLM2pcTest : public Test2PC
{
    void run(int party_id) override;
    void main_test(int party_id) override;
};

class OfflineCNN2pcTest : public Test2PC
{
    void main_test(int party_id) override;
};
