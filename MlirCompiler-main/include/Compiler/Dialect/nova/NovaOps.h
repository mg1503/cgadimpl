#ifndef NOVA_OPS_H
#define NOVA_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Compiler/Dialect/nova/NovaDialect.h"

#define GET_OP_CLASSES


#include "Compiler/Dialect/nova/NovaOpsEnums.h.inc"

#define GET_ATTR_CLASSES
#include "Compiler/Dialect/nova/NovaOpsAttributes.h.inc"

#include "Compiler/Dialect/nova/NovaOps.h.inc"


#endif // NOVA_OPS_H
