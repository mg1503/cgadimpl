//============================================================
// file: cgadimpl/include/ad/version_tracker.hpp
// PURPOSE: Shadow versioning system for in-place operation detection
//============================================================
#pragma once
#include <unordered_map>
#include <mutex>
#include "tensor.hpp"

namespace ag {
namespace detail {

/**
 * VersionTracker maintains a shadow map tracking version numbers for tensors
 * WITHOUT modifying the tensor library. Detects in-place modifications that
 * would corrupt autograd.
 */
class VersionTracker {
private:
    std::unordered_map<const void*, int> versions_;
    std::mutex mutex_;
    
public:
    /**
     * Get current version of a tensor (0 if not tracked)
     */
    int get_version(const Tensor& t) {
        std::lock_guard<std::mutex> lock(mutex_);
        const void* ptr = t.data();
        if (!ptr) return 0;
        
        auto it = versions_.find(ptr);
        return (it != versions_.end()) ? it->second : 0;
    }
    
    /**
     * Increment version counter (called on in-place operations)
     */
    void bump_version(const Tensor& t) {
        std::lock_guard<std::mutex> lock(mutex_);
        const void* ptr = t.data();
        if (!ptr) return;
        
        versions_[ptr]++;
    }
    
    /**
     * Start tracking a tensor (initialize to version 0)
     */
    void register_tensor(const Tensor& t) {
        std::lock_guard<std::mutex> lock(mutex_);
        const void* ptr = t.data();
        if (!ptr) return;
        
        // Always reset to 0
        versions_[ptr] = 0;
    }
    
    /**
     * Stop tracking a tensor (cleanup when tensor is destroyed)
     */
    void unregister_tensor(const Tensor& t) {
        std::lock_guard<std::mutex> lock(mutex_);
        versions_.erase(t.data());
    }
    
    /**
     * Check if tensor version matches expected (throws if mismatch)
     */
    void check_version(const Tensor& t, int expected_version, const char* op_name = "operation") {
        int current = get_version(t);
        if (current != expected_version) {
            throw std::runtime_error(
                std::string("RuntimeError: one of the variables needed for gradient computation ")
                + "in " + op_name + " has been modified by an inplace operation. "
                + "Expected version " + std::to_string(expected_version)
                + " but got version " + std::to_string(current));
        }
    }
};

/**
 * Global singleton instance
 */
inline VersionTracker& version_tracker() {
    static VersionTracker tracker;
    return tracker;
}

} // namespace detail
} // namespace ag
