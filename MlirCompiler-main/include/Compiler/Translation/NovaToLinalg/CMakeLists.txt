# CMakeLists.txt for Nova to linalg Lowering Pass

add_mlir_library(MLIRNovaToLinalgTransforms
  NovaToLinalg.cpp
  TNovaToLinalg.cpp
  NovaToLinalg.h

  DEPENDS
  MLIRLinalgDialect

  LINK_LIBS PUBLIC
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRNova
)

target_include_directories(MLIRNovaToLinalgTransforms PRIVATE
  ${PROJECT_SOURCE_DIR}/include/Compiler/Translation/NovaToLinalg
)