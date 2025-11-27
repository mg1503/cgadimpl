module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %matmul = transform.structured.match ops{["linalg.matmul", "linalg.generic"]} in %root 
      : (!transform.any_op) -> !transform.any_op

    // ========== L3 CACHE BLOCKING (MC × NC) ==========
    // Tile M (640) and N (1200) dimensions for L3 cache locality.
    %tiled_l3, %forall_l3 = transform.structured.tile_using_forall %matmul tile_sizes [640, 1200, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ========== L2 CACHE BLOCKING (KC) ==========  
    // Tile the K (reduction) dimension for L2 cache locality.
    %tiled_l2, %forall_l2 = transform.structured.tile_using_for %tiled_l3 tile_sizes [0, 0, 500]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ========== REGISTER BLOCKING (MR × NR) ==========
    // Tile M (16) and N (6) dimensions again for vector/register blocking.
    %tiled_reg, %for_m, %for_n = transform.structured.tile_using_for %tiled_l2 tile_sizes [16, 8, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // ========== VECTORIZATION ==========
    // Vectorize the innermost loops based on the register tile size (16x6).
    transform.structured.vectorize %tiled_reg : !transform.any_op

    transform.yield
  }
}
