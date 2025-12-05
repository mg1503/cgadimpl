
#include "ad/schema.hpp"
#include <iostream>
#include <cassert>
#include <string>

int main() {
    std::cout << "--- Running Schema Test ---" << std::endl;
    std::cout << "Total operations defined (OpCount): " << ag::OpCount << std::endl;
    std::cout << "--------------------------------" << std::endl;

    bool all_tests_passed = true;
    for (int i = 0; i < ag::OpCount; ++i) {
        ag::Op current_op = static_cast<ag::Op>(i);
        const char* name = ag::op_name(current_op);
        int arity = ag::op_arity(current_op);

        std::cout << "Op " << i << ": Name = '" << name << "', Arity = " << arity << std::endl;

        // Basic assertion: the name should not be null or empty.
        assert(name != nullptr && std::string(name).length() > 0 && "Operation name should not be empty.");
    }

    std::cout << "--------------------------------" << std::endl;
    std::cout << "Schema test completed successfully." << std::endl;
    return 0;
}
