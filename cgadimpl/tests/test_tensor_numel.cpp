#include "tensor.hpp"
#include <iostream>

using namespace ag;

int main() {
    Tensor t1;
    std::cout << "Tensor() numel: " << t1.numel() << "\n";
    
    // Check if reset exists and what it does
    // Note: Tensor might not have reset() if it's not a smart pointer wrapper directly exposed
    // But let's try to assign a default constructed tensor
    
    t1 = Tensor();
    std::cout << "After assign Tensor(): numel=" << t1.numel() << "\n";
    
    // Check if we can create empty tensor
    try {
        Tensor t2(Shape({0}), TensorOptions());
        std::cout << "Tensor({0}) numel: " << t2.numel() << "\n";
    } catch (const std::exception& e) {
        std::cout << "Tensor({0}) failed: " << e.what() << "\n";
    }
    
    return 0;
}
