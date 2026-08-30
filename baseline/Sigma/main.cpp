#include <iostream>
#include <cstring>
#include <functional>
#include <map>
#include "tests/test_base.h"

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <test_name> " << "<party_id>" << std::endl;
        return 1;
    }

    std::map<std::string, std::function<Test *()>> test_map = {
        // {"twopc", []()
        //  { return new TwoPcProtocolTest(); }},
        // {"bench", []()
        //  { return new TwoPcProtocolBenchmarkTest(); }},
        // {"cnn2pc", []()
        //  { return new CNN2pcTest(); }},
        {"llm2pc", []()
         { return new LLM2pcTest(); }},
        {"offline_llm2pc", []()
         { return new OfflineLLM2pcTest(); }},
        //  {"offline_cnn2pc", []()
        //  { return new OfflineCNN2pcTest(); }},
    };

    std::string test_name = argv[1];
    int party_id = std::stoi(argv[2]);

    if (test_map.find(test_name) != test_map.end())
    {
        Test *test = test_map[test_name]();
        test->run(party_id);
        delete test;
    }
    else
    {
        std::cerr << "Unknown test: " << test_name << std::endl;
        return 1;
    }

    return 0;
}