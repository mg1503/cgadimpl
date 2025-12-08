#include <stdlib.h>
#define N 512

void matmul(float A[N][N], float B[N][N], float C[N][N]) {
    // Initialize C to 0.0 (equivalent to linalg.fill)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0f;
        }
    }

    // Exact matmul (equivalent to linalg.matmul)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    // Create input tensors (equivalent to tensor.splat)
    float A[N][N];
    float B[N][N];
    float C[N][N];
    
    // Initialize A with 1.0 and B with 2.0
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = 1.0f;
            B[i][j] = 2.0f;
        }
    }
    // Call matmul
    matmul(A, B, C);
    
    // Verify result 
    float first_element = C[0][0];
    float expected = 1024.0f;  // 128 * 2.0
    
    // Return 0 if correct, 1 if wrong (exact same logic)
    if (first_element == expected) {
        return 0;
    } else {
        return 1;
    }
}