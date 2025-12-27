// // =====================
// // file: cgadimpl/include/ag/graph.hpp (declarations only)
// // =====================
// #pragma once
// #include <memory>
// #include <vector>
// #include <atomic>
// #include <mutex>

// #include "tensor.hpp"
// #include "ad/core/schema.hpp"
// #include "ad/runtime/runtime.hpp"

// namespace ag {
// struct Node;
// struct Value {
//     std::shared_ptr<Node> node;
//     Value();    
//     explicit Value(std::shared_ptr<Node> n);
//     const std::vector<int64_t>& shape() const;
//     std::pair<int, int> shape_2d() const;
//     Tensor& val();
//     const Tensor& val() const;
//     Tensor grad();
//     Tensor grad() const;
// };

// struct Node : std::enable_shared_from_this<Node> {
//     // Core tensors
//     Tensor tensor;    
    
//     // Graph structure
//     std::vector<std::shared_ptr<Node>> inputs;
//     std::vector<Value> saved_inputs;
//     std::vector<std::shared_ptr<Tensor>> tape;
    
//     // Checkpointing
//     std::vector<uint8_t> saved_rng_blob;
//     bool is_checkpoint{false};
//     bool has_saved_rng{false};
    
//     // Metadata
//     const char* debug_name{""};
//     Op op{Op::Leaf};
    
//     // Critical metadata
//     bool is_leaf{false};                    // Distinguishes parameters from computed values
//     std::vector<int> input_versions;        // Version tracking for in-place safety
    
//     // For Dependency Counter
//     std::atomic<int> child_grad_count{0};
//     std::mutex grad_mutex;

//     struct ExecutionContext {
//         ag_cuda_stream_t stream{nullptr};
//         DeviceIndex device;
//     };
//     ExecutionContext creation_context;      // Captured execution context
    
//     // bool requires_grad_flag_{false};
//     bool requires_grad() const { return tensor.requires_grad(); }
//     const std::vector<int64_t>& shape() const { return tensor.shape().dims; }
//     Node(const Tensor& v, Op op_, bool req_grad, const char* nm="");
//     Node() = default;
// };

// inline Value make_tensor(const Tensor& v, const char* name = "") {
//     return Value(std::make_shared<Node>(v, Op::Leaf, v.requires_grad(), name));
// }

// std::vector<Node*> topo_from(Node* root);
    
// } // namespace ag


// =====================
// file: cgadimpl/include/ad/core/graph.hpp
// =====================
// =====================
// file: cgadimpl/include/ad/core/graph.hpp
// =====================
#pragma once

#include "tensor.hpp"
#include "ad/core/schema.hpp"
#include <vector>
#include <memory>
#include <atomic>
#include <string>
#include <mutex>

namespace ag {

// Forward declarations to handle cyclic dependencies
struct Node;
struct Edge;

// ===========================================
// Edge: The Connection Logic (PyTorch Style)
// ===========================================
// Critical for LLaMA: Models with multiple outputs (like Split/Chunk)
// need to know WHICH output of a node flows into WHICH input of the next.
struct Edge {
    std::shared_ptr<Node> function; // The parent node
    uint32_t input_nr;              // Which output of the parent is this?
    
    bool is_valid() const { return function != nullptr; }
};

// ===========================================
// Value: The High-Level Variable Wrapper
// ===========================================
// This is your user-facing object. It decouples the "Variable" logic
// from the "Node" logic, allowing you to have multiple variables pointing
// to the same graph node (like Python references).
struct Value {
    std::shared_ptr<Node> node;
    uint32_t output_nr = 0; // Support for multi-output nodes
    
    Value() = default;
    
    // Core constructor
    explicit Value(std::shared_ptr<Node> n, uint32_t out_idx = 0) 
        : node(std::move(n)), output_nr(out_idx) {}
        
    // Public API - Declarations only (implementations below Node)
    OwnTensor::Tensor& val();
    const OwnTensor::Tensor& val() const;
    
    // Gradient Access
    OwnTensor::Tensor grad() const;
    
    // Shape/Dtype helpers
    const std::vector<int64_t>& shape() const;
    std::pair<int, int> shape_2d() const; // Restoring missing method if needed

    // Validity check
    bool defined() const { return node != nullptr; }
    
    // Operator overloads
    OwnTensor::Tensor* operator->();
};

// ===========================================
// Node: The Autograd Engine Unit
// ===========================================
struct Node : public std::enable_shared_from_this<Node> {
    
    std::vector<Edge> next_edges;  // 1. Connectivity & Graph Structure    
    OwnTensor::Tensor tensor;         // 2. Data Storage   
    Op op;         // 3. Metadata for execution
    std::string debug_name;
    uint64_t topological_nr = 0; // For deterministic execution order
    uint64_t sequence_nr = 0;    // For priority queue scheduling  
    std::atomic<int> child_grad_count{0};  // 4. Thread Safety & Parallelism
    mutable std::mutex mutex_;    
    uint32_t version_counter = 0;   // 5. Version Control (CRITICAL for In-Place Ops)
    std::vector<uint32_t> input_versions;

    // Checkpointing & Leaf Metadata
    bool is_checkpoint = false;
    bool is_leaf_flag = false; // We have a method is_leaf(), but some code might access member directly.

    // 6. Tape for Backward Pass (Context)
    // Stores intermediate tensors required for gradient computation (saved_tensors).
    // This replaces the old 'tape' mechanism.
    std::vector<std::shared_ptr<OwnTensor::Tensor>> tape;


    // Constructor (New API)
    Node(const OwnTensor::Tensor& t, Op op_type, const std::string& name = "")
        : tensor(t), op(op_type), debug_name(name) {        
        static std::atomic<uint64_t> global_seq_counter{0};
        sequence_nr = global_seq_counter++;
    }

    // Constructor (Legacy API for compatibility)
    Node(const OwnTensor::Tensor& t, Op op_type, bool /*requires_grad*/, const std::string& name = "")
        : Node(t, op_type, name) {}

    virtual ~Node() = default;

    // Adding a connection (thread-safe)
    void add_input(std::shared_ptr<Node> parent, uint32_t output_index = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        Edge e;
        e.function = parent;
        e.input_nr = output_index;
        next_edges.push_back(e);        
        // Update topology metadata
        if (parent && topological_nr <= parent->topological_nr) {
            topological_nr = parent->topological_nr + 1;
        }
    }

    // Helpers
    const std::vector<int64_t>& shape() const { return tensor.shape().dims; }

    // Checkpointing storage - Require Value to be defined
    std::vector<Value> saved_inputs;

    
    bool is_leaf() const { return op == Op::Leaf; }
    bool requires_grad() const { return tensor.requires_grad(); }
};

// ===========================================
// Value Implementation (Inline methods needing Node)
// ===========================================

inline OwnTensor::Tensor& Value::val() { return node->tensor; }
inline const OwnTensor::Tensor& Value::val() const { return node->tensor; }
// inline OwnTensor::Tensor Value::grad() const { return node->tensor.grad_view(); }
inline OwnTensor::Tensor Value::grad() const {
    if (!defined() || !node) return OwnTensor::Tensor(); // Safe fallback
    return node->tensor.grad_view();
}
inline const std::vector<int64_t>& Value::shape() const { return val().shape().dims; }
inline OwnTensor::Tensor* Value::operator->() { return &node->tensor; }

// ===========================================
// Factory Functions
// ===========================================
// Factory to create a leaf variable
inline Value make_tensor(const OwnTensor::Tensor& data, const std::string& name = "Leaf") {
    auto node = std::make_shared<Node>(data, Op::Leaf, name);
    return Value(node, 0);
}

// ===========================================
// Graph Algorithms
// ===========================================
// Use this for the traversal logic in your graph.cpp
std::vector<Node*> topo_from(Node* root);

} // namespace ag