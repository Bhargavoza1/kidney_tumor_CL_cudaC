 
#include "Tensor.h"
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <cassert>
namespace Hex {
    // Constructor
    template <typename T>
    Tensor<T>::Tensor(const std::vector<int>& shape, bool iscudafree) : shape(shape) ,_iscudafree(iscudafree) {
       
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        T* cudaData;
        cudaMalloc((void**)&cudaData, size * sizeof(T));
        data = std::shared_ptr<T[]>(cudaData, [=](T* ptr) { if (iscudafree) { cudaFree(ptr); } }); // Custom deleter for CUDA memory
    }

    // Destructor
    template <typename T>
    Tensor<T>::~Tensor() {
       //  cudafree();
    }
    template <typename T>
    void Tensor<T>::cudafree() { if (this != nullptr) { if (_iscudafree) { cudaFree(data.get()); }}}
 
 
    // Set element at index
    template <typename T>
    void Tensor<T>::set(const std::vector<int>& indices, T value) {
        int index = calculateIndex(indices);
        cudaMemcpy(data.get() + index, &value, sizeof(T), cudaMemcpyHostToDevice);
    }

    // Get element at index
    template <typename T>
    T Tensor<T>::get(const std::vector<int>& indices) const {
        int index = calculateIndex(indices);
        T value;
        cudaMemcpy(&value, data.get() + index, sizeof(T), cudaMemcpyDeviceToHost);
        return value;
    }

    // Print the tensor
    template <typename T>
    void Tensor<T>::print() const {
        std::cout << "Tensor (Shape: ";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) {
                std::cout << "x";
            }
        }
        std::cout << ", Type: " << typeid(T).name() << "):" << std::endl;

        printHelper(data.get(), shape, 0, {});
        //std::cout << std::endl;
    }

    template <typename T>
    void Tensor<T>::printshape() const {
        std::cout << "Tensor (Shape: ";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) {
                std::cout << "x";
            }
        }
        std::cout << ", Type: " << typeid(T).name() << "):" << std::endl;
    }

    template <typename T>
    void Tensor<T>::setData(T* newData) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        //  T* cudaData;
         // cudaMalloc((void**)&cudaData, size * sizeof(T));
         // data = std::shared_ptr<T[]>(cudaData, [=](T* ptr) { if (_iscudafree) { cudaFree(ptr); } }); // Custom deleter for CUDA memory
        cudaMemcpy(data.get(), newData, size * sizeof(T), cudaMemcpyDeviceToDevice);
    }

    // Getter for shape
    template <typename T>
    std::vector<int> Tensor<T>::getShape() const {
        return shape;
    }

    template <typename T>
    const T* Tensor<T>::getData() const {
        return data.get();
    }

    template <typename T>
    T* Tensor<T>::getData() {
        return data.get();
    }


    // Helper function to calculate the flat index from indices
    template <typename T>
    int Tensor<T>::calculateIndex(const std::vector<int>& indices) const {
        int index = 0;
        int stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            index += indices[i] * stride;
            stride *= shape[i];
        }
        return index;
    }

    // Helper function to print tensor data recursively
    template <typename T>
    void Tensor<T>::printHelper(const T* data, const std::vector<int>& shape, int dimension, std::vector<int> indices) const {
        int currentDimensionSize = shape[dimension];

        std::cout << "[";

        for (int i = 0; i < currentDimensionSize; ++i) {
            indices.push_back(i);

            if (dimension < shape.size() - 1) {
                // If not the last dimension, recursively print the next dimension
                printHelper(data, shape, dimension + 1, indices);
            }
            else {
                // If the last dimension, print the actual element
                std::cout << get(indices);
            }

            indices.pop_back();

            if (i < currentDimensionSize - 1) {
                std::cout << ", ";
            }
        }

        std::cout << "]";

        if (dimension < shape.size() - 1) {
            // If not the last dimension, add a new line after completing the inner block
            std::cout << std::endl;
        }
    }


    template <typename T>
    void Tensor<T>::reshape(const std::vector<int>& new_shape) {
        int new_size = 1;
        for (int dim : new_shape) {
            new_size *= dim;   
        }
 
        int size = 1;
        for (int dim : getShape() ) {
            size *= dim; 
        }


        if (new_size != size ) { 
            assert(false && "Error: New shape's total size does not match current size.");
            return;
        }
        shape = new_shape;
    }
     

    // Helper function to calculate indices from a flat index
     template <typename T>
     std::vector<int> Tensor<T>::calculateIndices(int index) const {
        std::vector<int> indices(shape.size(), 0);
        for (int i = shape.size() - 1; i >= 0; --i) {
            indices[i] = index % shape[i];
            index /= shape[i];
        }
        return indices;
     }
    // Other member function definitions...

    // Explicit instantiation of the template class for supported types
    template class Tensor<float>;
    template class Tensor<int>;
    template class Tensor<double>;
}
