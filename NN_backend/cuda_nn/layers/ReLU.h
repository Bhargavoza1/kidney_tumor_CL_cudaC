#pragma once
#include "layer.h"
#include <iostream>
 #include <memory>
namespace Hex{
	template <class T>
	class ReLU : public layer<T>
	{
    private:
 
        std::shared_ptr<Tensor<T>> input;
        std::unique_ptr<Tensor<T>> output;
        std::unique_ptr<Tensor<T>> input_error;

    public:
        // Constructor
        ReLU();
        ~ReLU();

        // Override forward method
        Tensor<T>& forward(Tensor<T>& input_tensor, bool Istraining = true) override;

        // Override backpropagation method
        Tensor<T>& backpropagation(Tensor<T>& output_error, float learning_rate = 0.001f) override;
 

    private:
        

    };

}



