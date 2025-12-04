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



void to_mlir( const std::string& input ){
  std::ifstream file(input);
  if(!file.is_open() ){
    std::cout<< "file is missing!!"<<std::endl;
  }

  std::cout<<"file found!! Reading contents..."<<std::endl;
  std::vector<std::string>declared_list;
  declared_list.push_back("matmul");
  declared_list.push_back("add");

  std::string line;
  while (std::getline(file, line)) {
    std::cout <<"reading line: "<< line << std::endl;
    bool match=false;
    for (const std::string& element : declared_list) {
       if(line.find(element)!=std::string::npos){
        std::cout<<"Operation found:  "<<element<<std::endl;
        match=true;

        break;
      }
      if (!match){
        continue;
          }
    }
    }



int main() {
  mlir::MLIRContext context;
  
  // Register the Nova dialect
  context.getOrLoadDialect<NovaDialect>();

  
  llvm::outs() << "Successfully loaded Nova dialect from installed mlir-compiler package!\n";
  llvm::outs() << "The mlir-compiler project is now reusable by other standalone projects.\n";
  
  return 0;
}
