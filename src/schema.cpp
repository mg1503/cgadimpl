#include "ad/schema.hpp"
#include <array>
// #include "ad/ops.def" // Ensure OpCount is defined or included

namespace ag {

const char* op_name(Op o) {
  static constexpr std::array<const char*, OpCount> names = {{
  #define OP(name, arity, str) str,
  #include "ad/ops.def"
  #undef OP
  }};
  return names[static_cast<std::size_t>(o)];
}

int op_arity(Op o) {
  static constexpr std::array<int, OpCount> arities = {{
  #define OP(name, arity, str) arity,
  #include "ad/ops.def"
  #undef OP
  }};
  return arities[static_cast<std::size_t>(o)];
}

} // namespace ag
