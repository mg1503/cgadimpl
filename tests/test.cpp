#include "ad/ag_all.hpp"
#include "ad/debug.hpp"
#include <iostream>
#include <iomanip>

static void printTensor(const char* name,
                        const ag::Tensor& T,
                        int max_r = -1, int max_c = -1,
                        int width = 9, int prec = 4) {
    using std::cout;
    using std::fixed;
    using std::setw;
    using std::setprecision;

    const int r = T.rows(), c = T.cols();
    if (max_r < 0) max_r = r;
    if (max_c < 0) max_c = c;

    cout << name << " [" << r << "x" << c << "]";
    if (r == 1 && c == 1) { // scalar fast path
        cout << " = " << fixed << setprecision(6) << T(0,0) << "\n";
        return;
    }
    cout << "\n";

    const int rr = std::min(r, max_r);
    const int cc = std::min(c, max_c);
    for (int i = 0; i < rr; ++i) {
        cout << "  ";
        for (int j = 0; j < cc; ++j) {
            cout << setw(width) << fixed << setprecision(prec) << T(i,j);
        }
        if (cc < c) cout << " ...";
        cout << "\n";
    }
    if (rr < r) cout << "  ...\n";
}

int main(){
    using namespace ag;
    using namespace std;
    // Tensor A = Tensor::randn(2,3);
    // Tensor B = Tensor::randn(3,2);
    auto a = param(Tensor::randn(2,3), "A");
    auto b = param(Tensor::randn(3,2), "B");
    auto bias = param(Tensor::zeros(1,2), "bias");
    auto y = sum(relu(matmul(a,b) + bias)); // scalar, tests
    zero_grad(y);
    backward(y);
    std::cout << "y = " << y.val().sum_scalar() << endl;
    cout << a.val() <<endl;
    return 0;
}
