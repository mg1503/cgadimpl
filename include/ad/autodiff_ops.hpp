#pragma once
#include <functional>
#include "ad/graph.hpp"
#include "ad/schema.hpp"
#include "ad/tensor.hpp"

namespace ag {

// VJP: given node n and its output upstream grad gy, accumulate grads into parents.
using VjpFn = void(*)(Node* n, const Tensor& gy);

// JVP: compute tangent for node n given a way to read parent tangents.
// tangent_of(p) must return the tangent T[p] (same shape as p->value).
using JvpFn = Tensor(*)(Node* n, const std::function<const Tensor&(Node*)>& tangent_of);

// Lookup tables (one slot per Op value).
VjpFn vjp_lookup(Op op);
JvpFn jvp_lookup(Op op);

} // namespace ag
