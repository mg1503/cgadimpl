#include "tensor.hpp" // Main adapter header
#include <iostream>
#include <cassert>
#include <vector> // Required for shape printing

// --- FIX: Modern print_tensor utility ---
static void print_tensor(const char* name, const Tensor& t) {
  std::cout << name << ": shape=[";
  const auto& dims = t.shape().dims;
  for(size_t i = 0; i < dims.size(); ++i) {
      std::cout << dims[i] << (i == dims.size() - 1 ? "" : ", ");
  }
  std::cout << "]  device=" << (t.is_cuda() ? "CUDA" : "CPU")
            << "  ptr=" << (const void*)t.data()
            << "\n";
}

using namespace OwnTensor; // For Shape, TensorOptions

int main() {
  // --- FIX: Use modern factories ---
  Tensor cpu = Tensor::ones(Shape{{2, 3}});
  Tensor gpu = Tensor::ones(Shape{{2, 3}}, TensorOptions().with_device(Device::CUDA));
  Tensor cpu2 = Tensor::zeros(cpu.shape(), ag::options(cpu));
  Tensor gpu2 =  Tensor::zeros(gpu.shape(), ag::options(gpu));

  print_tensor("cpu ", cpu);
  print_tensor("gpu ", gpu);
  print_tensor("cpu2", cpu2);
  print_tensor("gpu2", gpu2);

  // sanity checks (these are correct)
  assert( cpu.is_cpu() && !cpu.is_cuda());
  assert(!gpu.is_cpu() &&  gpu.is_cuda());
  assert(!gpu2.is_cpu() && gpu2.is_cuda());
  assert( cpu2.is_cpu() && !cpu2.is_cuda());

  std::cout << "[OK] device flags look correct\n";
  return 0;
}