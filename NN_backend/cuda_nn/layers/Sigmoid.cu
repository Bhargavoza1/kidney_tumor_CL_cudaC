#include "Sigmoid.h"
#include <cuda_runtime.h> 
#include <iostream>
namespace Hex {
    template<class T>
    Sigmoid<T>::Sigmoid() {}

    template<class T>
    Sigmoid<T>::~Sigmoid() {
       
    }
    
    template <typename T>
    __device__ T sigmoid(T x) {
        return static_cast<T>(1) / (static_cast<T>(1) + expf(-x));
    }
     
    template <typename T>
    __global__ void sigmoid_forward_kernel(const T* input, T* output, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            output[idx] = sigmoid(input[idx]);
        }
    }
     
    template<class T>
    Tensor<T>& Sigmoid<T>::forward(Tensor<T>& input_tensor, bool Istraining) {
        input = std::make_shared<Tensor<T>>(input_tensor);
        output.reset(new Tensor<T>(input_tensor.getShape()));

        std::vector<int> shape = input->getShape();
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
 
        sigmoid_forward_kernel << <(size + 255) / 256, 256 >> > (input->getData(), output->getData(), size);
        cudaDeviceSynchronize();
        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("error from sigmoid forward method : %s\n", cudaGetErrorString(cudaError));
            exit(EXIT_FAILURE);  // or handle the error appropriately
        }

        return *output;
    }



    template <typename T>
    __device__ T sigmoid_derivative(T x) {
        T sigmoid_value = sigmoid(x);
        return sigmoid_value * (static_cast<T>(1) - sigmoid_value);
    }

    // Kernel for backward pass using sigmoid activation
    template <typename T>
    __global__ void sigmoid_backward_kernel(const T* input, const T* output_error, T* output, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            output[idx] = sigmoid_derivative(input[idx]) * output_error[idx];
        }
    }

    // Update backpropagation method with sigmoid computation
    template<class T>
    Tensor<T>& Sigmoid<T>::backpropagation(Tensor<T>& output_error, float learning_rate) {
        input_error.reset(new Tensor<T>(output_error.getShape()));

        std::vector<int> shape = output_error.getShape();
     
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }

        sigmoid_backward_kernel << <(size + 255) / 256, 256 >> > (input->getData(), output_error.getData(), input_error->getData(), size);

        cudaDeviceSynchronize();
        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("error from sigmoid backward method : %s\n", cudaGetErrorString(cudaError));
            exit(EXIT_FAILURE);  // or handle the error appropriately
        }

        return *input_error;
    }

    template class Sigmoid<float>;
    template class Sigmoid<int>;
    template class Sigmoid<double>;
}