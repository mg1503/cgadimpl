// =====================
// file: cgadimpl/include/ag/schema.hpp (declarations only)
// =====================
#pragma once
#include <cstdint>

namespace ag {

enum class Op : uint8_t {
#define OP(name, arity, str) name,
#include "ad/detail/ops.def"
#undef OP
    Count
};

inline constexpr std::size_t OpCount = static_cast<std::size_t>(Op::Count);

const char* op_name(Op);
int         op_arity(Op);

} // namespace ag