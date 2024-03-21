#include "tensor_oprations.h"
#include "Tensor.h"
#include <curand_kernel.h>
#include <cuda_runtime.h> 
#include "Errorhelper.cpp"
#include <memory>
#include <iostream>
#include <vector>
 namespace Hex{ 
 
    template<class T, class U>
    __global__ void addKernel(const T* a, const U* b, typename std::common_type<T, U>::type* c, int size) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx < size) {
            c[idx] = static_cast<typename std::common_type<T, U>::type>(a[idx] + b[idx]);
        }
    }

    template<class T, class U>
    std::unique_ptr<Tensor<typename std::common_type<T, U>::type>> addTensor(const Tensor<T>& tensor1, const Tensor<U>& tensor2) {

        if (tensor1.getShape() != tensor2.getShape()) {
            std::cerr << "Error: Tensor shapes must be the same for addition. Shape of tensor1: "
                << shapeToString(tensor1.getShape()) << ", Shape of tensor2: " << shapeToString(tensor2.getShape()) << std::endl;
            exit(EXIT_FAILURE); // or use exit(EXIT_FAILURE) if you prefer
        }


        using CommonType = typename std::common_type<T, U>::type;
        std::vector<int> shape = tensor1.getShape();
        
        std::unique_ptr<Tensor<CommonType>> result(new Tensor<CommonType>(shape));

        
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }

        dim3 blockSize(256);
        dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
        addKernel<<<gridSize, blockSize>>> (tensor1.getData(), tensor2.getData(), result->getData(), size);
        cudaDeviceSynchronize();

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("CUDA error from add tensor: %s\n", cudaGetErrorString(cudaError));
            exit(EXIT_FAILURE);  // or handle the error appropriately
        }
        // Update the data pointer in the result tensor
       
        //result->setData(resultData);
 
        return result;
    }
 // CUDA kernel for tensor initialization with multiplication
    template <typename T>
    __global__ void initializeTensortoOne(T* data, int size ) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {


            data[index] = static_cast<T>(1.0f);
 
        }
    }

    template<typename T>
    void initTensorToOneOnGPU(Tensor<T>& tensor )
    {


        std::vector<int> shape = tensor.getShape();
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }

        // Launch CUDA kernel to initialize and multiply the tensor
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        initializeTensortoOne << <gridSize, blockSize >> > (tensor.getData(), size );
        cudaDeviceSynchronize();

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("CUDA error from init: %s\n", cudaGetErrorString(cudaError));
            exit(EXIT_FAILURE);  // or handle the error appropriately
        }

      
    }


  template <typename T>
    __device__ T customRandom(int seed, int index) {
        // Linear congruential generator parameters
        const int a = 16645;
        const int c = 10139;
        const int m = 21474; // 2^31

        // Update seed based on thread index
        seed = a * seed + c + index;

        // Generate pseudorandom number in [0, 1]
        int random_int = seed % m;
        T random_float = static_cast<T>(random_int) / static_cast<T>(m);


        return (random_float * (-static_cast<T>(0.5)));
    }

 

   // CUDA kernel for tensor initialization with multiplication
    template <typename T>
    __global__ void initializeTensor(T* data, int size, float multiplier) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {
           // data[index] = customRandom<T>(clock64(), index);

            curandState state;
            curand_init(777, index, 0, &state); // Initialize random number generator for each thread

            data[index] = curand_uniform(&state) * (2 * 0.5f) - 0.5f;


            //T value = static_cast<T>(index )  ;

            //if (multiplier != 0) {
            //    value *= multiplier;
            //}

            //data[index] = value;
        }
    }




   template<typename T>
    void initTensorOnGPU(Tensor<T>& tensor, float multiplier)
    {
        std::vector<int> shape = tensor.getShape();
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }

        // Launch CUDA kernel to initialize and multiply the tensor
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        initializeTensor << <gridSize, blockSize >> > (tensor.getData(), size, multiplier);
        cudaDeviceSynchronize();

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("CUDA error from init: %s\n", cudaGetErrorString(cudaError));
            exit(EXIT_FAILURE);  // or handle the error appropriately
        }
    }


  

    template <typename T>
    std::unique_ptr<Tensor<T>>  sliceFirstIndex(int firstIndex, const Tensor<T>& tensor) {
        if (firstIndex < 0 || firstIndex >= tensor.getShape()[0]) {
            throw std::out_of_range("Index out of range");
        }
        std::vector<int> shape = tensor.getShape();
        std::vector<int> newShape(shape.begin() + 1, shape.end()); // New shape without the first dimension
        std::unique_ptr<Tensor<T>> slicedTensor(new Tensor<T>(newShape));

        // Calculate offset for the first index
        int offset = firstIndex * shape[1] * shape[2]; // Assuming the shape is nxnxn

        // Copy data from original tensor to sliced tensor
        cudaMemcpy(slicedTensor->getData(), tensor.getData() + offset, newShape[0] * newShape[1] * sizeof(T), cudaMemcpyDeviceToDevice);

        return slicedTensor;
    }

    //template <typename T>
    //std::unique_ptr<Tensor<T>> slice(int index, Tensor<T> tensor)   {
    //    // Check if the index is within bounds
    //    if (index < 0 || index >= tensor.getShape()[0]) {
    //        throw std::out_of_range("Index out of bounds");
    //    }
    //    std::vector<int> shape = tensor.getShape();
    //    std::vector<int> sliced_shape(shape.begin() + 1, shape.end());
  
    //    std::unique_ptr<Tensor<T>> sliced_tensor(new Tensor<T>(sliced_shape));

    //    for (int i = 0; i < sliced_shape[0]; ++i) {
    //        std::vector<int> original_indices = { index, i };
    //        std::vector<int> sliced_indices = { i };
    //        for (size_t j = 1; j < shape.size(); ++j) {
    //            original_indices.push_back(0);
    //            sliced_indices.push_back(j - 1);
    //        }

    //        T value = tensor.get(original_indices);
    //        sliced_tensor->set(sliced_indices, value);
    //    }

    //    return sliced_tensor;
    //}
 

    template<typename T>
    __global__ void transpose_kernel(const T* input, T* output, int rows, int cols) {
        int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
        int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

        if (tid_x < cols && tid_y < rows) {
            output[tid_x * rows + tid_y] = input[tid_y * cols + tid_x];
        }
    }

    template <typename T>
    std::unique_ptr<Tensor<T>>  transpose(const Tensor<T>& tensor) {
        // Get the shape of the original tensor
        std::vector<int> original_shape = tensor.getShape();
      

        // Swap the dimensions
        std::vector<int> transposed_shape(original_shape.rbegin(), original_shape.rend());

        std::unique_ptr<Tensor<T>> transposed_tensor(new Tensor<T>(transposed_shape));

        dim3 threadsPerBlock(16, 16); // 16x16 threads per block
        dim3 numBlocks((original_shape[0] + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (original_shape[1] + threadsPerBlock.y - 1) / threadsPerBlock.y); // Adjust grid dimensions

        transpose_kernel<<<numBlocks, threadsPerBlock>>>(tensor.getData(), transposed_tensor->getData(), original_shape[0], original_shape[1]);

        return transposed_tensor;
    }
    	
 
   
	template void initTensorToOneOnGPU(Tensor<float>& tensor );
	template void initTensorOnGPU(Tensor<float>& tensor, float multiplier);
	template std::unique_ptr<Tensor<float>> transpose(const Tensor<float>& tensor);
	template std::unique_ptr<Tensor<float>>  sliceFirstIndex(int firstIndex, const Tensor<float>& tensor);

    template void initTensorToOneOnGPU(Tensor<int>& tensor );
    template void initTensorOnGPU(Tensor<int>& tensor, float multiplier);
	template std::unique_ptr<Tensor<int>> transpose(const Tensor<int>& tensor);
	template std::unique_ptr<Tensor<int>>  sliceFirstIndex(int firstIndex, const Tensor<int>& tensor);

    template void initTensorToOneOnGPU(Tensor<double>& tensor );
    template void initTensorOnGPU(Tensor<double>& tensor, float multiplier);
	template std::unique_ptr<Tensor<double>> transpose(const Tensor<double>& tensor);
	template std::unique_ptr<Tensor<double>>  sliceFirstIndex(int firstIndex, const Tensor<double>& tensor);

 

    template std::unique_ptr<Tensor<typename std::common_type<int, int>::type>> addTensor(const Tensor<int>& tensor1, const Tensor<int>& tensor2);
    template std::unique_ptr<Tensor<typename std::common_type<int, float>::type>> addTensor(const Tensor<int>& tensor1, const Tensor<float>& tensor2);
    template std::unique_ptr<Tensor<typename std::common_type<int, double>::type>> addTensor(const Tensor<int>& tensor1, const Tensor<double>& tensor2);

    template std::unique_ptr<Tensor<typename std::common_type<float, int>::type>> addTensor(const Tensor<float>& tensor1, const Tensor<int>& tensor2);
    template std::unique_ptr<Tensor<typename std::common_type<float, float>::type>> addTensor(const Tensor<float>& tensor1, const Tensor<float>& tensor2);
    template std::unique_ptr<Tensor<typename std::common_type<float, double>::type>> addTensor(const Tensor<float>& tensor1, const Tensor<double>& tensor2);

    template std::unique_ptr<Tensor<typename std::common_type<double, int>::type>> addTensor(const Tensor<double>& tensor1, const Tensor<int>& tensor2);
    template std::unique_ptr<Tensor<typename std::common_type<double, float>::type>> addTensor(const Tensor<double>& tensor1, const Tensor<float>& tensor2);
    template std::unique_ptr<Tensor<typename std::common_type<double, double>::type>> addTensor(const Tensor<double>& tensor1, const Tensor<double>& tensor2);

}
