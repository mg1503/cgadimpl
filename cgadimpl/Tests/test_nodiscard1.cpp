#include "ad/core/nodiscard.hpp"
#include "ad/core/graph.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <vector>

/**
 * @file test_nodiscard1.cpp
 * @brief Comprehensive test suite for AG_NODISCARD macro.
 * 
 * This file contains various test cases to ensure that the AG_NODISCARD macro
 * is correctly applied and recognized by the compiler across different scenarios:
 * 1. Basic return types (int, float, etc.)
 * 2. Library-specific types (std::shared_ptr<Node>)
 * 3. Struct-level nodiscard (Value struct)
 * 4. Template functions
 * 5. Class member functions
 * 6. Lambda functions
 * 7. Nested return scenarios
 */

// Helper to suppress unused variable warnings when we intentionally capture the result
template <typename T>
void use(const T& v) {
    (void)v;
}

// ============================================================================
// 1. Basic Return Types
// ============================================================================

// Test Case: Function returning a basic integer with AG_NODISCARD.
// Rationale: Ensures simple scalar returns are protected.
AG_NODISCARD int get_important_code() {
    return 42;
}

// Test Case: Function returning a double with AG_NODISCARD.
AG_NODISCARD double get_precision_value() {
    return 3.14159265359;
}

// ============================================================================
// 2. Library-Specific Types (Simulating nodeops.hpp)
// ============================================================================

// Test Case: Function returning a shared_ptr to a Node.
// Rationale: In autograd systems, forgetting to use a newly created node
// usually means a disconnected graph or a memory leak of logic.
AG_NODISCARD std::shared_ptr<ag::Node> create_op_node() {
    return std::make_shared<ag::Node>();
}

// ============================================================================
// 3. Struct-Level Nodiscard (Simulating graph.hpp)
// ============================================================================

// Test Case: Function returning a type that is itself marked AG_NODISCARD.
// Rationale: If the struct 'Value' is marked AG_NODISCARD, any function
// returning it should trigger a warning if the result is ignored.
AG_NODISCARD ag::Value compute_value() {
    return ag::Value(1.0f);
}

// ============================================================================
// 4. Template Functions
// ============================================================================

// Test Case: Template function with AG_NODISCARD.
// Rationale: Ensures the macro works correctly during template instantiation.
template <typename T>
AG_NODISCARD T identity_check(T val) {
    return val;
}

// ============================================================================
// 5. Class Member Functions
// ============================================================================

class Calculator {
public:
    // Test Case: Const member function with AG_NODISCARD.
    AG_NODISCARD int add(int a, int b) const {
        return a + b;
    }

    // Test Case: Static member function with AG_NODISCARD.
    AG_NODISCARD static float pi() {
        return 3.14f;
    }
};

// ============================================================================
// 6. Lambda Functions (C++17 and later)
// ============================================================================

// Note: AG_NODISCARD on lambdas requires specific syntax depending on compiler support.
// Typically: []() AG_NODISCARD { return 1; }
auto nodiscard_lambda = []() -> int {
    return 100;
};

// ============================================================================
// 7. Nested Scenarios
// ============================================================================

// Test Case: A function that calls another nodiscard function and returns its result.
AG_NODISCARD int nested_call() {
    return get_important_code();
}

// ============================================================================
// Main Execution
// ============================================================================

int main() {
    std::cout << "Running Nodiscard Tests..." << std::endl;

    // --- Correct Usage (Capturing results) ---
    
    int code = get_important_code();
    use(code);

    double d = get_precision_value();
    use(d);

    auto node = create_op_node();
    use(node);

    ag::Value val = compute_value();
    use(val);

    int t_val = identity_check<int>(10);
    use(t_val);

    Calculator calc;
    int sum = calc.add(5, 5);
    use(sum);

    float p = Calculator::pi();
    use(p);

    int l_val = nodiscard_lambda();
    use(l_val);

    int n_val = nested_call();
    use(n_val);

    // --- Intentional Ignored Results (For compiler warning verification) ---
    // IMPORTANT: These will trigger COMPILER WARNINGS during build.
    // They do NOT affect runtime output, but you will see warnings in the console
    // if you run the build command (e.g., 'make test_nodiscard1').
    
    get_important_code();    // Should warn: ignoring return value
    get_precision_value();   // Should warn: ignoring return value
    create_op_node();        // Should warn: ignoring return value
    compute_value();         // Should warn: ignoring return value
    identity_check(20);      // Should warn: ignoring return value
    calc.add(1, 2);          // Should warn: ignoring return value
    Calculator::pi();        // Should warn: ignoring return value
    nested_call();           // Should warn: ignoring return value

    std::cout << "Nodiscard Tests Completed Successfully (Results Captured)." << std::endl;
    return 0;
}
