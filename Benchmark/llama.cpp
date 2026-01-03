#include <torch/torch.h>  
#include <vector>  
  
class RMSNorm : public torch::nn::Module {  
public:  
    RMSNorm(int64_t hidden_size, float eps = 1e-6f)  
        : eps_(eps),  
          weight_(register_parameter("weight", torch::ones(hidden_size))) {}  
  
    torch::Tensor forward(const torch::Tensor& x) {  
        auto variance = x.pow(2).mean(-1, true);  
        auto x_normed = x * torch::rsqrt(variance + eps_);  
        return x_normed * weight_;  
    }  
  
private:  
    float eps_;  
    torch::Tensor weight_;  
};  
  
class RotaryEmbedding : public torch::nn::Module {  
public:  
    RotaryEmbedding(int64_t dim, int64_t max_position_embeddings = 2048, float base = 10000.0f)  
        : dim_(dim), base_(base) {  
          
        auto inv_freq = torch::arange(0, dim, 2).to(torch::kFloat) / dim;  
        inv_freq = 1.0f / (pow(base_, inv_freq));  
        inv_freq_ = register_buffer("inv_freq", inv_freq);
    }  
  
    std::pair<torch::Tensor, torch::Tensor> forward(  
        const torch::Tensor& x,   
        int64_t seq_len = 0) {  
        
        if (seq_len == 0) {  
            seq_len = x.size(-2);  
        }  
        
        auto t = torch::arange(seq_len, x.options()).unsqueeze(-1);  
        auto freqs = torch::matmul(t, inv_freq_.unsqueeze(0));  
        
        // Compute cos and sin directly from freqs  
        auto cos = torch::cos(freqs);  
        auto sin = torch::sin(freqs);  
        
        return {cos, sin};  
    }
  
    static torch::Tensor apply_rotary_pos_emb(  
        const torch::Tensor& q,  
        const torch::Tensor& k,  
        const torch::Tensor& cos,  
        const torch::Tensor& sin,  
        const torch::Tensor& position_ids) {  
        
        auto cos_ = cos.index({position_ids}).unsqueeze(1);  
        auto sin_ = sin.index({position_ids}).unsqueeze(1);  
        
        // Split q and k into two halves  
        auto q1 = q.index({"...", torch::indexing::Slice(0, q.size(-1) / 2)});  
        auto q2 = q.index({"...", torch::indexing::Slice(q.size(-1) / 2, torch::indexing::None)});  
        auto k1 = k.index({"...", torch::indexing::Slice(0, k.size(-1) / 2)});  
        auto k2 = k.index({"...", torch::indexing::Slice(k.size(-1) / 2, torch::indexing::None)});  
        
        // Apply rotation to first half only  
        auto q_rot = (q1 * cos_) + (rotate_half(q1) * sin_);  
        auto k_rot = (k1 * cos_) + (rotate_half(k1) * sin_);  
        
        // Concatenate rotated first half with unrotated second half  
        auto q_embed = torch::cat({q_rot, q2}, -1);  
        auto k_embed = torch::cat({k_rot, k2}, -1);  
        
        return torch::cat({q_embed, k_embed}, -1);  
    }
  
private:  
    int64_t dim_;  
    float base_;  
    torch::Tensor inv_freq_;  
  
    static torch::Tensor rotate_half(const torch::Tensor& x) {  
        auto x1 = x.index({"...", torch::indexing::Slice(0, x.size(-1) / 2)});  
        auto x2 = x.index({"...", torch::indexing::Slice(x.size(-1) / 2, torch::indexing::None)});  
        return torch::cat({-x2, x1}, -1);  
    }  
};  
  
class Attention : public torch::nn::Module {  
public:  
    Attention(int64_t hidden_size, int64_t num_heads, int64_t num_kv_heads, float max_position_embeddings = 2048.0f)  
        : hidden_size_(hidden_size),  
          num_heads_(num_heads),  
          num_kv_heads_(num_kv_heads),  
          num_groups_(num_heads / num_kv_heads),  
          head_dim_(hidden_size / num_heads),  
          scaling_(1.0f / std::sqrt(head_dim_)) {  
  
        q_proj_ = register_module("q_proj", torch::nn::Linear(hidden_size, num_heads * head_dim_));  
        k_proj_ = register_module("k_proj", torch::nn::Linear(hidden_size, num_kv_heads * head_dim_));  
        v_proj_ = register_module("v_proj", torch::nn::Linear(hidden_size, num_kv_heads * head_dim_));  
        o_proj_ = register_module("o_proj", torch::nn::Linear(num_heads * head_dim_, hidden_size));  
          
        rotary_emb_ = register_module("rotary_emb",   
            std::make_shared<RotaryEmbedding>(head_dim_, static_cast<int64_t>(max_position_embeddings)));  
    }  
  
    static torch::Tensor repeat_kv(const torch::Tensor& x, int64_t n_rep) {  
        auto sizes = x.sizes();  
        auto bs = sizes[0];  
        auto n_kv_heads = sizes[1];  
        auto slen = sizes[2];  
        auto head_dim = sizes[3];  
          
        if (n_rep == 1) return x;  
          
        std::vector<int64_t> expand_shape = {bs, n_kv_heads, slen, n_rep, head_dim};  
        return x.unsqueeze(3).expand(expand_shape)  
                   .reshape({bs, n_kv_heads * n_rep, slen, head_dim});  
    }  
  
    torch::Tensor forward(  
        const torch::Tensor& x,  
        const torch::Tensor& attention_mask,  
        const torch::Tensor& position_ids,  
        bool use_cache = false,  
        const torch::Tensor& past_key_value = torch::Tensor()) {  
          
        auto sizes = x.sizes();  
        auto bsz = sizes[0];  
        auto q_len = sizes[1];  
          
        auto q = q_proj_->forward(x);  
        auto k = k_proj_->forward(x);  
        auto v = v_proj_->forward(x);  
          
        std::vector<int64_t> view_shape = {bsz, q_len, num_heads_, head_dim_};  
        q = q.view(view_shape).transpose(1, 2);  
          
        std::vector<int64_t> kv_view_shape = {bsz, q_len, num_kv_heads_, head_dim_};  
        k = k.view(kv_view_shape).transpose(1, 2);  
        v = v.view(kv_view_shape).transpose(1, 2);  
          
        auto [cos, sin] = rotary_emb_->forward(k, q_len);  
          
        q = RotaryEmbedding::apply_rotary_pos_emb(q, k, cos, sin, position_ids);  
        k = RotaryEmbedding::apply_rotary_pos_emb(k, k, cos, sin, position_ids);  
          
        if (use_cache && past_key_value.defined()) {  
            k = torch::cat({past_key_value[0], k}, -2);  
            v = torch::cat({past_key_value[1], v}, -2);  
        }  
          
        auto past_kv = torch::Tensor();  
        if (use_cache) {  
            past_kv = torch::stack({k, v});  
        }  
          
        k = repeat_kv(k, num_groups_);  
        v = repeat_kv(v, num_groups_);  
          
        auto attn_weights = torch::matmul(q, k.transpose(-2, -1)) * scaling_;  
          
        if (attention_mask.defined()) {  
            attn_weights = attn_weights + attention_mask;  
        }  
          
        attn_weights = torch::softmax(attn_weights, -1);  
        auto attn_output = torch::matmul(attn_weights, v);  
          
        attn_output = attn_output.transpose(1, 2).contiguous();  
        attn_output = attn_output.view({bsz, q_len, -1});  
          
        return o_proj_->forward(attn_output);  
    }  
  
private:  
    int64_t hidden_size_;  
    int64_t num_heads_;  
    int64_t num_kv_heads_;  
    int64_t num_groups_;  
    int64_t head_dim_;  
    float scaling_;  
      
    torch::nn::Linear q_proj_{nullptr}, k_proj_{nullptr}, v_proj_{nullptr}, o_proj_{nullptr};  
    std::shared_ptr<RotaryEmbedding> rotary_emb_;  
};  
  
class FeedForward : public torch::nn::Module {  
public:  
    FeedForward(int64_t hidden_size, int64_t intermediate_size)  
        : gate_proj_(register_module("gate_proj", torch::nn::Linear(hidden_size, intermediate_size))),  
          up_proj_(register_module("up_proj", torch::nn::Linear(hidden_size, intermediate_size))),  
          down_proj_(register_module("down_proj", torch::nn::Linear(intermediate_size, hidden_size))) {}  
  
    torch::Tensor forward(const torch::Tensor& x) {  
        auto gate = torch::nn::functional::silu(gate_proj_->forward(x));  
        return down_proj_->forward(gate * up_proj_->forward(x));  
    }  
  
private:  
    torch::nn::Linear gate_proj_{nullptr}, up_proj_{nullptr}, down_proj_{nullptr};  
};  
  
class LlamaBlock : public torch::nn::Module {  
public:  
    LlamaBlock(int64_t hidden_size, int64_t num_heads, int64_t num_kv_heads,   
               int64_t intermediate_size, float max_position_embeddings = 2048.0f)  
        : attention_1_(register_module("attention_1",   
            std::make_shared<Attention>(hidden_size, num_heads, num_kv_heads, max_position_embeddings))),  
          mlp_(register_module("mlp", std::make_shared<FeedForward>(hidden_size, intermediate_size))),  
          input_layernorm_(register_module("input_layernorm",   
              std::make_shared<RMSNorm>(hidden_size))),  
          post_attention_layernorm_(register_module("post_attention_layernorm",   
              std::make_shared<RMSNorm>(hidden_size))) {}  
  
    torch::Tensor forward(  
        const torch::Tensor& x,  
        const torch::Tensor& attention_mask,  
        const torch::Tensor& position_ids,  
        bool use_cache = false,  
        const torch::Tensor& past_key_value = torch::Tensor()) {  
          
        auto residual = x;  
        auto hidden_states = input_layernorm_->forward(x);  
          
        hidden_states = attention_1_->forward(  
            hidden_states, attention_mask, position_ids, use_cache, past_key_value);  
        hidden_states = residual + hidden_states;  
          
        residual = hidden_states;  
        hidden_states = post_attention_layernorm_->forward(hidden_states);  
        hidden_states = mlp_->forward(hidden_states);  
        hidden_states = residual + hidden_states;  
          
        return hidden_states;  
    }  
  
private:  
    std::shared_ptr<Attention> attention_1_;  
    std::shared_ptr<FeedForward> mlp_;  
    std::shared_ptr<RMSNorm> input_layernorm_;  
    std::shared_ptr<RMSNorm> post_attention_layernorm_;  
};  
  
class Llama2Model : public torch::nn::Module {  
public:  
    Llama2Model(  
        int64_t vocab_size = 32000,  
        int64_t hidden_size = 4096,  
        int64_t num_layers = 32,  
        int64_t num_heads = 32,  
        int64_t num_kv_heads = 32,  
        int64_t intermediate_size = 11008,  
        float max_position_embeddings = 2048.0f)  
        : vocab_size_(vocab_size),  
          hidden_size_(hidden_size),  
          num_layers_(num_layers) {  
  
        embed_tokens_ = register_module("embed_tokens",   
            torch::nn::Embedding(vocab_size, hidden_size));  
        norm_ = register_module("norm", std::make_shared<RMSNorm>(hidden_size));  
          
        layers_ = register_module("layers", torch::nn::ModuleList());  
        for (int i = 0; i < num_layers; ++i) {  
            layers_->push_back(std::make_shared<LlamaBlock>(  
                hidden_size, num_heads, num_kv_heads, intermediate_size, max_position_embeddings));  
        }  
          
        lm_head_ =  register_module("lm_head", torch::nn::Linear(torch::nn::LinearOptions(hidden_size, vocab_size).bias(false)));
    }  
  
    torch::Tensor forward(  
        const torch::Tensor& input_ids,  
        const torch::Tensor& attention_mask = torch::Tensor(),  
        const torch::Tensor& position_ids = torch::Tensor(),  
        bool use_cache = false) {  
          
        auto inputs_embeds = embed_tokens_->forward(input_ids);  
          
        torch::Tensor pos_ids;  
        if (!position_ids.defined()) {  
            pos_ids = torch::arange(input_ids.size(1), input_ids.options())  
                           .unsqueeze(0)  
                           .expand_as(input_ids);  
        } else {  
            pos_ids = position_ids;  
        }  
          
        auto hidden_states = inputs_embeds;  
          
        for (auto& layer : *layers_) {  
            hidden_states = layer->as<LlamaBlock>()->forward(  
                hidden_states, attention_mask, pos_ids, use_cache);  
        }  
          
        hidden_states = norm_->forward(hidden_states);  
        auto logits = lm_head_->forward(hidden_states);  
          
        return logits;  
    }  
  
private:  
    int64_t vocab_size_;  
    int64_t hidden_size_;  
    int64_t num_layers_;  
      
    torch::nn::Embedding embed_tokens_{nullptr};  
    torch::nn::ModuleList layers_{nullptr};  
    std::shared_ptr<RMSNorm> norm_;  
    torch::nn::Linear lm_head_{nullptr};  
};  
  
int main() {  
    // Set default dtype to float16  
    torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kFloat16)); 
      
     auto model = std::make_shared<Llama2Model>(  
        32000,  // vocab_size (unchanged)  
        2048,   // hidden_size (reduced from 4096)  
        16,     // num_layers (reduced from 32)  
        16,     // num_heads (reduced from 32)  
        16,     // num_kv_heads (reduced from 32)  
        5504,   // intermediate_size (reduced from 11008)  
        2048.0f // max_position_embeddings  
    );  
      
    // Move model to GPU and convert to half precision  
    model->to(torch::kCUDA);  
    model->to(torch::kFloat16);  
      
    auto input_ids = torch::tensor({{1, 2, 3, 4, 5}}, torch::kLong).to(torch::kCUDA);  
    auto attention_mask = torch::ones_like(input_ids);  
      
    auto logits = model->forward(input_ids, attention_mask);  
      
    std::cout << "Output logits shape: " << logits.sizes() << std::endl;  
      
    return 0;  
}