#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Inline helper function for clean, C++ style error checking
inline void checkCuda(cudaError_t result, const std::string& msg) {
    if (result != cudaSuccess) {
        throw std::runtime_error(msg + ": " + cudaGetErrorString(result));
    }
}

// Function prototype using references to std::vector
void addWithCuda(std::vector<int>& c, const std::vector<int>& a, const std::vector<int>& b);

__global__ void addKernel(int *c, const int *a, const int *b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    const int arraySize = 5;
    
    // Using std::vector instead of raw C-style arrays
    std::vector<int> a = { 1, 2, 3, 4, 5 };
    std::vector<int> b = { 10, 20, 30, 40, 50 };
    std::vector<int> c(arraySize, 0);

    try {
        // Add vectors in parallel
        addWithCuda(c, a, b);

        // Using std::cout instead of printf
        std::cout << "{1,2,3,4,5} + {10,20,30,40,50} = {"
                  << c[0] << "," << c[1] << "," << c[2] << "," 
                  << c[3] << "," << c[4] << "}\n";
                  
    } catch (const std::exception& e) {
        // Using std::cerr instead of fprintf
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return 1;
    }

    // cudaDeviceReset must be called before exiting
    if (cudaDeviceReset() != cudaSuccess) {
        std::cerr << "cudaDeviceReset failed!" << std::endl;
        return 1;
    }

    return 0;
}

void addWithCuda(std::vector<int>& c, const std::vector<int>& a, const std::vector<int>& b) {
    int *dev_a = nullptr;
    int *dev_b = nullptr;
    int *dev_c = nullptr;
    unsigned int size = a.size();

    // Choose which GPU to run on
    checkCuda(cudaSetDevice(0), "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");

    // Use a try-catch block to handle GPU memory cleanup safely without using 'goto'
    try {
        // Allocate GPU buffers
        checkCuda(cudaMalloc((void**)&dev_c, size * sizeof(int)), "cudaMalloc dev_c failed");
        checkCuda(cudaMalloc((void**)&dev_a, size * sizeof(int)), "cudaMalloc dev_a failed");
        checkCuda(cudaMalloc((void**)&dev_b, size * sizeof(int)), "cudaMalloc dev_b failed");

        // Copy input vectors from host memory to GPU buffers (.data() gets the raw pointer from std::vector)
        checkCuda(cudaMemcpy(dev_a, a.data(), size * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy dev_a failed");
        checkCuda(cudaMemcpy(dev_b, b.data(), size * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy dev_b failed");

        // Launch a kernel on the GPU with one thread for each element
        addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

        // Check for any errors launching the kernel
        checkCuda(cudaGetLastError(), "addKernel launch failed");
        
        // cudaDeviceSynchronize waits for the kernel to finish
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

        // Copy output vector from GPU buffer to host memory
        checkCuda(cudaMemcpy(c.data(), dev_c, size * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy dev_c failed");

    } catch (...) {
        // If anything fails above, free the memory and re-throw the exception to main()
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);
        throw; 
    }

    // If everything succeeds, free memory normally
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
}