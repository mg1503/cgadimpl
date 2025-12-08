#ifndef NOVA_DIALECT
#define NOVA_DIALECT

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// Nova Dialect
//===----------------------------------------------------------------------===//


def Nova_Dialect : Dialect {
  let name = "nova";
  let summary = "A high-level dialect for machine learning operations";
  let description = [{
    The Nova dialect provides a high-level representation for ML operations.
  }];
  let cppNamespace = "::mlir::nova";
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 0;

  let usePropertiesForAttributes = 0;
}
#endif // NOVA_DIALECT
