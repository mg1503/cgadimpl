# CMakeLists.txt for Nova to Arith Lowering Pass

add_mlir_library(MLIRNovaToArithTransforms
  NovaToArith.cpp
  NovaToArith.h

  DEPENDS
  MLIRArithDialect

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRNova
)

target_include_directories(MLIRNovaToArithTransforms PRIVATE
  ${PROJECT_SOURCE_DIR}/include/Compiler/Translation/NovaToArith
)