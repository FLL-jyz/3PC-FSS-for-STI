#ifndef LLMCONFIG_H
#define LLMCONFIG_H
#include <cstdint>
#include <stdexcept>
#include <string>

class TransformerConfig
{
public:
    uint32_t seq_len, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, num_layers, vocab_size;
    bool isReLUAct;

    TransformerConfig() = default;

    TransformerConfig(uint32_t seq_len, uint32_t hidden_size, uint32_t intermediate_size, uint32_t num_attention_heads,
                      uint32_t num_layers, bool isReLUAct, uint32_t vocab_size = 0, uint32_t num_key_value_heads = 0) : seq_len(seq_len), hidden_size(hidden_size),
                                                             intermediate_size(intermediate_size),
                                                             num_attention_heads(num_attention_heads),
                                                             num_key_value_heads(num_key_value_heads == 0 ? num_attention_heads : num_key_value_heads),
                                                             num_layers(num_layers), vocab_size(vocab_size), isReLUAct(isReLUAct)
    {
    }

    TransformerConfig(const std::string &model_type, const int seq_len) : seq_len(seq_len)
    {
        if (model_type == "bert-base" || model_type == "gpt2")
        {
            hidden_size = 768;
            intermediate_size = 3072;
            num_attention_heads = 12;
            num_key_value_heads = num_attention_heads;
            num_layers = 12;
            vocab_size = (model_type == "bert-base") ? 30522 : 50257;
            isReLUAct = false;
        }
        else if (model_type == "bert-large")
        {
            hidden_size = 1024;
            intermediate_size = 4096;
            num_attention_heads = 16;
            num_key_value_heads = num_attention_heads;
            num_layers = 24;
            vocab_size = 30522;
            isReLUAct = false;
        }
        else if (model_type == "gpt2-XL")
        {
            hidden_size = 1600;
            intermediate_size = 6400;
            num_attention_heads = 25;
            num_key_value_heads = num_attention_heads;
            num_layers = 48;
            vocab_size = 50257;
            isReLUAct = false;
        }
        else if (model_type == "transformer")
        {
            hidden_size = 512;
            intermediate_size = 2048;
            num_attention_heads = 8;
            num_key_value_heads = num_attention_heads;
            num_layers = 6;
            vocab_size = 30522;
            isReLUAct = true;
        }
        else if (model_type == "llama2" || model_type == "llama2-7b")
        {
            hidden_size = 4096;
            intermediate_size = 11008;
            num_attention_heads = 32;
            num_key_value_heads = 32;
            num_layers = 32;
            vocab_size = 32000;
            isReLUAct = false;
        }
        else if (model_type == "llama2-13b")
        {
            hidden_size = 5120;
            intermediate_size = 13824;
            num_attention_heads = 40;
            num_key_value_heads = 40;
            num_layers = 40;
            vocab_size = 32000;
            isReLUAct = false;
        }
        else if (model_type == "llama2-70b")
        {
            hidden_size = 8192;
            intermediate_size = 28672;
            num_attention_heads = 64;
            num_key_value_heads = 8;
            num_layers = 80;
            vocab_size = 32000;
            isReLUAct = false;
        }
        else
        {
            throw std::invalid_argument("Invalid model type");
        }
    }
};

#endif // LLMCONFIG_H
