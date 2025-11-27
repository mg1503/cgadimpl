#include "Compiler/Dialect/nova/NovaDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <iostream>
#include <fstream>

using namespace mlir;
using namespace nova;

string input = "path/to/file/test.ir";


void to_mlir( std::string& input ){
  std::ifstream file(input);
  if(!file.is_open() ){
    std::cout<< "file is missing!!"<<std::endl;
  }

  std::cout<<"file found!!"<<std::endll;
  file.close();

  std::string line;
  while (std::getline(input, line)) {
    std::cout << line << endl;
  }

  input.close();
}



int main() {
  mlir::MLIRContext context;
  
  // Register the Nova dialect
  context.getOrLoadDialect<NovaDialect>();

  
  llvm::outs() << "Successfully loaded Nova dialect from installed mlir-compiler package!\n";
  llvm::outs() << "The mlir-compiler project is now reusable by other standalone projects.\n";
  
  return 0;
}
