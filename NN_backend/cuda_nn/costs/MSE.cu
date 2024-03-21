 
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "../utils/Tensor.h"
#include <memory>
 namespace Hex{
    template<typename T>
    __global__ void mse_mean_kernel(const T* y_true, const T* y_pred, T* result, int batch_size, int feature_size, int size) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;


        if (row < batch_size && col < feature_size) {
            int index = row * feature_size + col;
            T diff = y_true[index] - y_pred[index];
            atomicAdd(result, diff * diff);
        }
        if (row == 0 && col == 0) {
            // Calculate mean
            *result = *result / size;
        }
        __syncthreads();
    }

    template<typename T>
    std::unique_ptr<Tensor<T>> mse(Tensor<T>& y_true, Tensor<T>& y_pred) {
        std::unique_ptr<Tensor<T>> result(new Tensor<T>({ 1 }));

        std::vector<int> shape = y_true.getShape();
        int batch_size = shape[0];
        int feature_size = shape[1];
        int size = batch_size * feature_size;

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x, (feature_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

        mse_mean_kernel << <numBlocks, threadsPerBlock >> > (y_true.getData(), y_pred.getData(), result->getData(), batch_size, feature_size, size);
        cudaDeviceSynchronize();
        return result;
    }

    template<typename T>
    __global__ void mse_derivative_kernel(const T* y_true, const T* y_pred, T* derivative, int batch_size, int feature_size, int size) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;


        if (row < batch_size && col < feature_size) {
            int index = row * feature_size + col;
            derivative[index] = (2.0 / static_cast<T>(size)) * (y_pred[index] - y_true[index]);
        }
    }

    template<typename T>
    std::unique_ptr<Tensor<T>> mse_derivative(Tensor<T>& y_true, Tensor<T>& y_pred) {
        std::unique_ptr<Tensor<T>> derivative(new Tensor<T>(y_true.getShape()));

        std::vector<int> shape = y_true.getShape();
        int batch_size = shape[0];
        int feature_size = shape[1];
        int size = batch_size * feature_size;

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x, (feature_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

        mse_derivative_kernel << <numBlocks, threadsPerBlock >> > (y_true.getData(), y_pred.getData(), derivative->getData(), batch_size, feature_size, size);
        cudaDeviceSynchronize();
        return derivative;
    }
 
  
     template std::unique_ptr<Tensor<float>> mse(Tensor<float>& y_true, Tensor<float>& y_pred);

   
     template std::unique_ptr<Tensor<float>> mse_derivative(Tensor<float>& y_true, Tensor<float>& y_pred);

          template std::unique_ptr<Tensor<int>> mse(Tensor<int>& y_true, Tensor<int>& y_pred);

   
     template std::unique_ptr<Tensor<int>> mse_derivative(Tensor<int>& y_true, Tensor<int>& y_pred);

          template std::unique_ptr<Tensor<double>> mse(Tensor<double>& y_true, Tensor<double>& y_pred);

   
     template std::unique_ptr<Tensor<double>> mse_derivative(Tensor<double>& y_true, Tensor<double>& y_pred);
}