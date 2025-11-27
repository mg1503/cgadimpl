module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_parallelize(%root: !transform.any_op) {
    %forall_ops = transform.structured.match ops{["scf.forall"]} in %root
      : (!transform.any_op) -> !transform.any_op
    
    transform.foreach %forall_ops : !transform.any_op {
    ^bb0(%forall_op: !transform.any_op):
      transform.loop.forall_to_parallel %forall_op : (!transform.any_op) -> !transform.any_op
      transform.yield
    }

    transform.yield
  }
}
