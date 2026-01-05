#ifndef PLAIN_LAYERS_H
#define PLAIN_LAYERS_H

#include "tensor.h"
#include "llmConfig.h"
#include <omp.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <map>
#include <string>

namespace plain {

template<int B, int S>
struct FixedPoint {
    static constexpr int bits = B;
    static constexpr int scale = S;
};

template<typename T>
struct PrecisionTrait {
    static constexpr int bits = 64;
    static constexpr int scale = 16;
};

template<int B, int S>
struct PrecisionTrait<FixedPoint<B, S>> {
    static constexpr int bits = B;
    static constexpr int scale = S;
};

template <typename T>
void simulate_fixed_point(Tensor<T>& t, int bits, int scale) {
    if constexpr (std::is_floating_point_v<T>) {
        T max_val = std::pow(2, bits - 1) - 1;
        T min_val = -std::pow(2, bits - 1);
        T factor = std::pow(2, scale);
        
        #pragma omp parallel for
        for(size_t i=0; i<t.size(); ++i) {
            T val = t.data[i];
            val = std::round(val * factor);
            if (val > max_val) val = max_val;
            if (val < min_val) val = min_val;
            t.data[i] = val / factor;
        }
    }
}

template <typename T>
T simulate_fixed_point_scalar(T val, int bits, int scale) {
    if constexpr (std::is_floating_point_v<T>) {
        T max_val = std::pow(2, bits - 1) - 1;
        T min_val = -std::pow(2, bits - 1);
        T factor = std::pow(2, scale);
        
        val = std::round(val * factor);
        if (val > max_val) val = max_val;
        if (val < min_val) val = min_val;
        return val / factor;
    }
    return val;
}

template <typename P, typename T>
void truncate(Tensor<T>& t) {
    constexpr int bits = PrecisionTrait<P>::bits;
    constexpr int scale = PrecisionTrait<P>::scale;
    simulate_fixed_point(t, bits, scale);
}

template <typename P, typename T>
T truncate_scalar(T val) {
    constexpr int bits = PrecisionTrait<P>::bits;
    constexpr int scale = PrecisionTrait<P>::scale;
    return simulate_fixed_point_scalar(val, bits, scale);
}

template <typename T, typename U = T, typename V = U>
class Layer {
public:
    std::string name;
    std::map<std::string, Tensor<T>> parameters;
    virtual ~Layer() = default;
    virtual Tensor<T> forward(Tensor<T>& x) { return x; }
    virtual void load_param(const std::map<std::string, Tensor<T>>& params, const std::string& prefix) {}
};

template <typename T, typename U = T, typename V = U>
class Linear : public Layer<T, U, V> {
public:
    Tensor<T> weight;
    Tensor<T> bias;
    uint32_t in_features, out_features;

    Linear(uint32_t in, uint32_t out, std::string name_ = "Linear") : in_features(in), out_features(out) {
        this->name = name_;
        weight.allocate({in, out});
        bias.allocate({out});
        weight.zero();
        bias.zero();
        this->parameters["weight"] = weight;
        this->parameters["bias"] = bias;
    }

    void load_param(const std::map<std::string, Tensor<T>>& params, const std::string& prefix) override {
        std::string w_key = prefix + "weight";
        std::string b_key = prefix + "bias";
        if (params.count(w_key)) {
            Tensor<T> loaded_w = params.at(w_key);
            if (loaded_w.shape == weight.shape) {
                weight = loaded_w;
            } else if (loaded_w.shape.size() == 2 && weight.shape.size() == 2 && 
                       loaded_w.shape[0] == weight.shape[1] && loaded_w.shape[1] == weight.shape[0]) {
                // Transpose if shape matches transpose (e.g. loading [out, in] to [in, out])
                weight = loaded_w.transpose();
            } else {
                std::cerr << "Warning: Shape mismatch for " << w_key << ". Expected " 
                          << weight.shape[0] << "x" << weight.shape[1] << ", got " 
                          << loaded_w.shape[0] << "x" << loaded_w.shape[1] << std::endl;
            }
            this->parameters["weight"] = weight;
        }
        if (params.count(b_key)) {
            bias = params.at(b_key);
            this->parameters["bias"] = bias;
        }
    }

    Tensor<T> forward(Tensor<T>& x) override {
        // x: [..., in]
        // Reshape to 2D for matmul: [batch_dims, in]
        std::vector<uint32_t> original_shape = x.shape;
        uint32_t batch_size = 1;
        for(size_t i=0; i<original_shape.size()-1; ++i) batch_size *= original_shape[i];
        
        Tensor<T> x_flat = x;
        x_flat.reshape({batch_size, in_features});
        
        Tensor<T> out;
        out.allocate({batch_size, out_features});
        Tensor<T>::matmul(out, x_flat, weight);
        
        // Add bias
        #pragma omp parallel for collapse(2)
        for(uint32_t i=0; i<batch_size; ++i) {
            for(uint32_t j=0; j<out_features; ++j) {
                out.data[i*out_features + j] += bias.data[j];
            }
        }
        
        std::vector<uint32_t> out_shape = original_shape;
        out_shape.back() = out_features;
        out.reshape(out_shape);
        truncate<T>(out);
        return out;
    }
};

template <typename T, typename U = T, typename V = U>
class LayerNorm : public Layer<T, U, V> {
public:
    Tensor<T> weight;
    Tensor<T> bias;
    uint32_t normalized_shape;
    T eps = 1e-5;

    LayerNorm(uint32_t shape, std::string name_ = "LayerNorm") : normalized_shape(shape) {
        this->name = name_;
        weight.allocate({shape});
        bias.allocate({shape});
        weight.ones();
        bias.zero();
        this->parameters["weight"] = weight;
        this->parameters["bias"] = bias;
    }

    void load_param(const std::map<std::string, Tensor<T>>& params, const std::string& prefix) override {
        std::string w_key = prefix + "weight";
        std::string b_key = prefix + "bias";
        if (params.count(w_key)) {
            weight = params.at(w_key);
            this->parameters["weight"] = weight;
        }
        if (params.count(b_key)) {
            bias = params.at(b_key);
            this->parameters["bias"] = bias;
        }
    }

    Tensor<T> forward(Tensor<T>& x) override {
        truncate<V>(x);
        Tensor<T> out(x.shape);
        uint32_t hidden = x.shape.back();
        uint32_t n = x.size() / hidden;
        
        for(uint32_t i=0; i<n; ++i) {
            T sum = 0;
            T sum_sq = 0;
            #pragma omp parallel for reduction(+:sum,sum_sq)
            for(uint32_t j=0; j<hidden; ++j) {
                T val = x.data[i*hidden + j];
                sum += val;
                sum_sq += val * val;
            }
            T mean = sum / hidden;
            T var = sum_sq / hidden - mean * mean;
            T inv_std = 1.0 / std::sqrt(var + eps);
            inv_std = truncate_scalar<U>(inv_std);
            
            #pragma omp parallel for
            for(uint32_t j=0; j<hidden; ++j) {
                out.data[i*hidden + j] = (x.data[i*hidden + j] - mean) * inv_std * weight.data[j] + bias.data[j];
            }
        }
        truncate<V>(out);
        return out;
    }
};

template <typename T, typename U = T, typename V = U>
class GELU : public Layer<T, U, V> {
public:
    Tensor<T> forward(Tensor<T>& x) override {
        truncate<V>(x);
        Tensor<T> out(x.shape);
        #pragma omp parallel for
        for(size_t i=0; i<x.size(); ++i) {
            T val = x.data[i];
            T c = 0.7978845608; 
            out.data[i] = 0.5 * val * (1.0 + std::tanh(c * (val + 0.044715 * val * val * val)));
        }
        truncate<V>(out);
        return out;
    }
};

template <typename T, typename U = T, typename V = U>
class Softmax : public Layer<T, U, V> {
public:
    Tensor<T> forward(Tensor<T>& x) override {
        truncate<V>(x);
        Tensor<T> out(x.shape);
        uint32_t last_dim = x.shape.back();
        uint32_t n = x.size() / last_dim;
        
        for(uint32_t i=0; i<n; ++i) {
            T max_val = -1e9;
            #pragma omp parallel for reduction(max:max_val)
            for(uint32_t j=0; j<last_dim; ++j) {
                if(x.data[i*last_dim + j] > max_val) max_val = x.data[i*last_dim + j];
            }
            
            T sum_exp = 0;
            #pragma omp parallel for reduction(+:sum_exp)
            for(uint32_t j=0; j<last_dim; ++j) {
                T val = std::exp(x.data[i*last_dim + j] - max_val);
                out.data[i*last_dim + j] = val;
                sum_exp += val;
            }
            
            #pragma omp parallel for
            for(uint32_t j=0; j<last_dim; ++j) {
                out.data[i*last_dim + j] /= sum_exp;
            }
        }
        truncate<V>(out);
        return out;
    }
};

template <typename T, typename U = T, typename V = U>
class BertSelfAttention : public Layer<T, U, V> {
public:
    Linear<T, U, V> query, key, value;
    Linear<T, U, V> dense; // Output projection
    uint32_t num_heads, head_size;
    Softmax<T, U, V> softmax;
    
    BertSelfAttention(TransformerConfig& config, std::string name_ = "BertSelfAttention") 
        : query(config.hidden_size, config.hidden_size, "query"),
          key(config.hidden_size, config.hidden_size, "key"),
          value(config.hidden_size, config.hidden_size, "value"),
          dense(config.hidden_size, config.hidden_size, "dense"),
          num_heads(config.num_attention_heads),
          head_size(config.hidden_size / config.num_attention_heads) {
              this->name = name_;
          }

    void load_param(const std::map<std::string, Tensor<T>>& params, const std::string& prefix) override {
        query.load_param(params, prefix + "query.");
        key.load_param(params, prefix + "key.");
        value.load_param(params, prefix + "value.");
        dense.load_param(params, prefix + "dense.");
    }
          
    Tensor<T> forward(Tensor<T>& x) override {
        // x: [B, S, H]
        uint32_t B = x.shape[0];
        uint32_t S = x.shape[1];
        
        Tensor<T> Q = query.forward(x);
        Tensor<T> K = key.forward(x);
        Tensor<T> V_tensor = value.forward(x);
        
        // Reshape and permute: [B, S, N, D] -> [B, N, S, D]
        Q.reshape({B, S, num_heads, head_size});
        Q = Q.permute({0, 2, 1, 3});
        
        K.reshape({B, S, num_heads, head_size});
        K = K.permute({0, 2, 1, 3});
        
        V_tensor.reshape({B, S, num_heads, head_size});
        V_tensor = V_tensor.permute({0, 2, 1, 3});
        
        // Scores = Q @ K.T
        Tensor<T> K_T = K.transpose(); // [B, N, D, S]
        Tensor<T> scores({B, num_heads, S, S});
        Tensor<T>::matmul(scores, Q, K_T);
        
        // Scale
        Tensor<T>::div(scores, std::sqrt((T)head_size));
        
        truncate<V>(scores);

        // Softmax
        Tensor<T> probs = softmax.forward(scores);
        truncate<T>(probs);
        
        // Out = probs @ V
        Tensor<T> context({B, num_heads, S, head_size});
        Tensor<T>::matmul(context, probs, V_tensor);
        
        // Permute back: [B, S, N, D]
        context = context.permute({0, 2, 1, 3});
        context.reshape({B, S, num_heads * head_size});
        Tensor<T> output = dense.forward(context);
        return output;
    }
};

template <typename T, typename U = T, typename V = U>
class BertLayer : public Layer<T, U, V> {
public:
    BertSelfAttention<T, U, V> attention;
    LayerNorm<T, U, V> ln1, ln2;
    Linear<T, U, V> ffn1, ffn2;
    GELU<T, U, V> act;
    
    BertLayer(TransformerConfig& config, std::string name_ = "BertLayer") 
        : attention(config, "attention"),
          ln1(config.hidden_size, "ln1"), ln2(config.hidden_size, "ln2"),
          ffn1(config.hidden_size, config.intermediate_size, "ffn1"),
          ffn2(config.intermediate_size, config.hidden_size, "ffn2") {
              this->name = name_;
          }

    void load_param(const std::map<std::string, Tensor<T>>& params, const std::string& prefix) override {
        attention.load_param(params, prefix + "attention.");
        ln1.load_param(params, prefix + "ln1.");
        ln2.load_param(params, prefix + "ln2.");
        ffn1.load_param(params, prefix + "ffn1.");
        ffn2.load_param(params, prefix + "ffn2.");
    }
          
    Tensor<T> forward(Tensor<T>& x) override {
        Tensor<T> residual = x;
        Tensor<T> a = attention.forward(x);
        Tensor<T>::add(a, a, residual);
        Tensor<T> ln1_out = ln1.forward(a);
        
        residual = ln1_out;
        Tensor<T> f = ffn1.forward(ln1_out);
        truncate<V>(f);
        f = act.forward(f);
        truncate<T>(f);
        f = ffn2.forward(f);
        Tensor<T>::add(f, f, residual);
        Tensor<T> out = ln2.forward(f);
        return out;
    }
};

template <typename T, typename U = T, typename V = U>
class BertEncoder : public Layer<T, U, V> {
public:
    std::vector<BertLayer<T, U, V>*> layers;
    
    BertEncoder(TransformerConfig& config, std::string name_ = "BertEncoder") {
        this->name = name_;
        for(int i=0; i<config.num_layers; ++i) {
            layers.push_back(new BertLayer<T, U, V>(config, "layer." + std::to_string(i)));
        }
    }
    
    ~BertEncoder() {
        for(auto l : layers) delete l;
    }

    void load_param(const std::map<std::string, Tensor<T>>& params, const std::string& prefix) override {
        for(size_t i=0; i<layers.size(); ++i) {
            layers[i]->load_param(params, prefix + "layer." + std::to_string(i) + ".");
        }
    }
    
    Tensor<T> forward(Tensor<T>& x) override {
        Tensor<T> out = x;
        for(auto l : layers) {
            out = l->forward(out);
        }
        return out;
    }
};

template <typename T, typename U = T, typename V = U>
class Bert : public Layer<T, U, V> {
public:
    LayerNorm<T, U, V> first_ln;
    BertEncoder<T, U, V> encoder;

    Bert(TransformerConfig& config, std::string name_ = "Bert") 
        : first_ln(config.hidden_size, "first_ln"),
          encoder(config, "encoder") {
        this->name = name_;
    }

    void load_param(const std::map<std::string, Tensor<T>>& params, const std::string& prefix) override {
        first_ln.load_param(params, prefix + "embeddings.LayerNorm."); // Assuming standard Bert structure often has embeddings LN
        encoder.load_param(params, prefix + "encoder.");
    }

    Tensor<T> forward(Tensor<T>& x) override {
        Tensor<T> ln_out = first_ln.forward(x);
        return encoder.forward(ln_out);
    }
};

template <typename T, typename U = T, typename V = U>
class GPT2Attention : public Layer<T, U, V> {
public:
    Linear<T, U, V> c_attn;
    Linear<T, U, V> c_proj;
    uint32_t num_heads, head_size, hidden_size;
    Softmax<T, U, V> softmax;
    
    GPT2Attention(TransformerConfig& config, std::string name_ = "GPT2Attention") 
        : c_attn(config.hidden_size, config.hidden_size * 3, "c_attn"),
          c_proj(config.hidden_size, config.hidden_size, "c_proj"),
          num_heads(config.num_attention_heads),
          head_size(config.hidden_size / config.num_attention_heads),
          hidden_size(config.hidden_size) {
              this->name = name_;
          }

    void load_param(const std::map<std::string, Tensor<T>>& params, const std::string& prefix) override {
        c_attn.load_param(params, prefix + "c_attn.");
        c_proj.load_param(params, prefix + "c_proj.");
    }
          
    Tensor<T> forward(Tensor<T>& x) override {
        uint32_t B = x.shape[0];
        uint32_t S = x.shape[1];
        
        Tensor<T> qkv = c_attn.forward(x);
        // Split Q, K, V
        // qkv: [B, S, 3*H]
        // We need to split last dim into 3 parts
        
        Tensor<T> Q({B, S, hidden_size});
        Tensor<T> K({B, S, hidden_size});
        Tensor<T> V_tensor({B, S, hidden_size});
        
        // Manual split
        #pragma omp parallel for collapse(2)
        for(uint32_t i=0; i<B*S; ++i) {
            for(uint32_t j=0; j<hidden_size; ++j) {
                Q.data[i*hidden_size + j] = qkv.data[i*3*hidden_size + j];
                K.data[i*hidden_size + j] = qkv.data[i*3*hidden_size + hidden_size + j];
                V_tensor.data[i*hidden_size + j] = qkv.data[i*3*hidden_size + 2*hidden_size + j];
            }
        }
        
        Q.reshape({B, S, num_heads, head_size});
        Q = Q.permute({0, 2, 1, 3});
        
        K.reshape({B, S, num_heads, head_size});
        K = K.permute({0, 2, 1, 3});
        
        V_tensor.reshape({B, S, num_heads, head_size});
        V_tensor = V_tensor.permute({0, 2, 1, 3});
        
        Tensor<T> K_T = K.transpose();
        Tensor<T> scores({B, num_heads, S, S});
        Tensor<T>::matmul(scores, Q, K_T);
        Tensor<T>::div(scores, std::sqrt((T)head_size));
        
        truncate<V>(scores);

        // Masking for GPT2 (causal)
        // Lower triangular mask
        #pragma omp parallel for collapse(3)
        for(uint32_t b=0; b<B; ++b) {
            for(uint32_t h=0; h<num_heads; ++h) {
                for(uint32_t i=0; i<S; ++i) {
                    for(uint32_t j=i+1; j<S; ++j) {
                        scores.data[b*num_heads*S*S + h*S*S + i*S + j] = -1e9;
                    }
                }
            }
        }
        
        Tensor<T> probs = softmax.forward(scores);
        truncate<T>(probs);
        Tensor<T> context({B, num_heads, S, head_size});
        Tensor<T>::matmul(context, probs, V_tensor);
        
        context = context.permute({0, 2, 1, 3});
        context.reshape({B, S, hidden_size});
        
        truncate<T>(context);
        

        Tensor<T> output = c_proj.forward(context);
        return output;
    }
};

template <typename T, typename U = T, typename V = U>
class GPT2Layer : public Layer<T, U, V> {
public:
    GPT2Attention<T, U, V> attn;
    LayerNorm<T, U, V> ln1, ln2;
    Linear<T, U, V> ffn1, ffn2;
    GELU<T, U, V> act;
    
    GPT2Layer(TransformerConfig& config, std::string name_ = "GPT2Layer") 
        : attn(config, "attn"),
          ln1(config.hidden_size, "ln1"), ln2(config.hidden_size, "ln2"),
          ffn1(config.hidden_size, config.intermediate_size, "ffn1"), // GPT2 usually 4*H
          ffn2(config.intermediate_size, config.hidden_size, "ffn2") {
              this->name = name_;
          }

    void load_param(const std::map<std::string, Tensor<T>>& params, const std::string& prefix) override {
        attn.load_param(params, prefix + "attn.");
        ln1.load_param(params, prefix + "ln1.");
        ln2.load_param(params, prefix + "ln2.");
        ffn1.load_param(params, prefix + "ffn1.");
        ffn2.load_param(params, prefix + "ffn2.");
    }
          
    Tensor<T> forward(Tensor<T>& x) override {
        Tensor<T> residual = x;
        Tensor<T> ln1_out = ln1.forward(x);
        Tensor<T> a = attn.forward(ln1_out);
        Tensor<T>::add(a, a, residual);
        
        residual = a;
        Tensor<T> ln2_out = ln2.forward(a);
        Tensor<T> f = ffn1.forward(ln2_out);
        truncate<V>(f);
        f = act.forward(f);
        truncate<T>(f);
        f = ffn2.forward(f);
        Tensor<T>::add(f, f, residual);
        Tensor<T> out = f;
        return out;
    }
};

template <typename T, typename U = T, typename V = U>
class GPT2 : public Layer<T, U, V> {
public:
    std::vector<GPT2Layer<T, U, V>*> layers;
    LayerNorm<T, U, V> ln_f;
    
    GPT2(TransformerConfig& config, std::string name_ = "GPT2") : ln_f(config.hidden_size, "ln_f") {
        this->name = name_;
        for(int i=0; i<config.num_layers; ++i) {
            layers.push_back(new GPT2Layer<T, U, V>(config, "h." + std::to_string(i)));
        }
    }
    
    ~GPT2() {
        for(auto l : layers) delete l;
    }

    void load_param(const std::map<std::string, Tensor<T>>& params, const std::string& prefix) override {
        for(size_t i=0; i<layers.size(); ++i) {
            layers[i]->load_param(params, prefix + "h." + std::to_string(i) + ".");
        }
        ln_f.load_param(params, prefix + "ln_f.");
    }
    
    Tensor<T> forward(Tensor<T>& x) override {
        Tensor<T> out = x;
        for(auto l : layers) {
            out = l->forward(out);
        }
        return ln_f.forward(out);
    }
};

} // namespace plain

#endif // PLAIN_LAYERS_H
