go to github: https://github.com/mg1503/cgadimpl/tree/experiment
download as zip file and open in suitable code editor

delete the tensor folder first and then ctrl + j ,
in terminal now run the following:
    git init
    git config --global init.defaultBranch main
    git submodule add --force https://github.com/kathir-23s/Tensor-Implementations tensor

once the cloning of the tensor library is done do the following:
    cd tensor
    git checkout bb569ca198d10b60f69072b56ada8704f0a0158a

then apply the following changes to the files mentioned:

tensor/src/core/Tensor.cpp: ( 2 updates)

    1) nbytes:

    size_t Tensor::nbytes() const 
        {
            // return numel() * size_t(dtype_); // data_size_
            return numel() * dtype_size(dtype_);    ---> this is the change
        }

    2) copy_

    Tensor& Tensor::copy_(const Tensor& src)
        {
            // Edge case: Self-copy is no-op
            if (this == &src || data() == src.data()) return *this;
            // Edge case: Empty tensor
            if (numel() == 0 && src.numel() == 0) {
                return *this;
            }
            // Edge case: Size validation
            if (numel() != src.numel()) {
                throw std::runtime_error(
                    "copy_: size mismatch. Destination has " + 
                    std::to_string(numel()) + " elements but source has " + 
                    std::to_string(src.numel())
                );
            }
            if (dtype_ != src.dtype_) {
                throw std::runtime_error("copy_: dtype mismatch");
            }
            if (numel() == 0) return *this;
            if (!is_contiguous() || storage_offset_ != 0) {
                throw std::runtime_error("copy_: destination must be contiguous");
            }
            
            // Materialize non-contiguous source
            const Tensor* src_ptr = &src;
            Tensor src_contig;
            if (!src.is_contiguous() ) {
                src_contig = src.contiguous();
                src_ptr = &src_contig;
            }
            try {
                device::copy_memory(
                    // data(), device_.device,           // destination ptr and device
                    // src_ptr->data(), src_ptr->device_.device,  // source ptr and device
                    // nbytes()
                    this->data(), 
                    this->device_.device,           // destination ptr and device
                    src_ptr->data(), 
                    src_ptr->device_.device,  // source ptr and device
                    src_ptr->nbytes()
                );
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("copy_ failed: ") + e.what());
            }
            
            return *this;
        }


tensor/src/core/TensorUtils.cpp:
void Tensor::display(std::ostream& os, int precision) const {
    PrintOptions opts;
    opts.precision = precision;

    // 1. Create a CPU copy of the main tensor's data to print safely.
    //    The .to() method will handle all cases (CPU->CPU, CUDA->CPU).
    Tensor data_to_print = this->to(Device::CPU);

    // 2. Print the header using the ORIGINAL tensor's metadata (*this)
    os << "Tensor(shape=(";
    for (size_t i = 0; i < shape_.dims.size(); ++i) {
        os << shape_.dims[i] << (i + 1 < shape_.dims.size() ? ", " : "");
    }
    os << "), dtype=" << get_dtype_name(dtype_) << ", device='";
    if (device_.device == Device::CPU) {
        os << "cpu";
    } else {
        os << "cuda:" << device_.index;
    }
    os << "'";
    if (requires_grad_) os << ", requires_grad=True";
    os << ")\n";

    // 3. Print the data using the SAFE CPU COPY.
    if (data_to_print.numel() == 0) {
        os << "[]\n";
    } else {
        std::vector<int64_t> idx;
        idx.reserve(data_to_print.shape().dims.size());
        // Call the recursive printer on the CPU copy and its CPU data pointer.
        print_recursive_from_base(os, data_to_print, data_to_print.data(), idx, 0, opts);
        os << "\n";
    }

    // 4. Do the same intelligent copy-then-print for the gradient.
    if (requires_grad_ && grad_ptr_) {
        os << "\nGrad(shape=(";
        // ... (print Grad header as before) ...
        os << "')\n";

        // Create a temporary Tensor object that WRAPS the grad_ptr_
        Tensor grad_tensor(grad_ptr_, shape_, stride_, 0, dtype_, device_, false);
        // Now, create a safe CPU copy of that gradient tensor.
        Tensor grad_to_print = grad_tensor.to(Device::CPU);
        
        if (grad_to_print.numel() == 0) {
            os << "[]\n";
        } else {
            std::vector<int64_t> idx;
            idx.reserve(grad_to_print.shape().dims.size());
            print_recursive_from_base(os, grad_to_print, grad_to_print.data(), idx, 0, opts);
            os << "\n";
        }
    }
}

tensor make:
NVCCFLAGS = -std=c++20 -Xcompiler="-fPIC" -arch=sm_86 -g --extended-lambda


in tensor.h : (line 74)
 //✨✨✨ 
        Tensor(Shape shape, bool requires_grad = false)   //----> make it false
        : Tensor(shape, Dtype::Float32, DeviceIndex(Device::CPU), requires_grad=false) {}

        Tensor() = default;


now run the file using the command from the root folder

bash ztools/run.sh

if this fails do these:

for "/usr/bin/ld: cannot find -ltbb " error try this:
    sudo apt install libtbb-dev

if cmake is not installed install it by running this code:
    sudo apt install cmake

if you want to check existing cmake run:
    cmake --version

make sure it is >3.2 series