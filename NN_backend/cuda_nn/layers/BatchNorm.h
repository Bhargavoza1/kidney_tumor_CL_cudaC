#pragma once
 
#include "layer.h"
#include <iostream>
#include <memory>

#define MAX_CHANNELS 64

namespace Hex {


    enum class TensorShape {
        _2D,
        _4D
    };

    template <class T>
    class BatchNorm : public layer<T>
    {
    private:
        
        TensorShape _Tshape;

        float momentum;
        float eps;

        
        std::shared_ptr<Tensor<T>> gamma;  
        std::shared_ptr<Tensor<T>> beta;  
        std::shared_ptr<Tensor<T>> running_mean;
        std::shared_ptr<Tensor<T>> running_var;

        std::shared_ptr<Tensor<T>> input;
     
        Tensor<T> input_mean;
        Tensor<T> input_var;

    


        std::unique_ptr<Tensor<T>> x_normalized;
        std::unique_ptr<Tensor<T>> input_error;
        std::unique_ptr<Tensor<T>> grad_normalized;

        std::unique_ptr<Tensor<T>> output;

    

    public:
        BatchNorm(int features_or_channels,TensorShape tensorshape = TensorShape::_4D ,float momentum = 0.9, float eps = 1e-5);
        
        ~BatchNorm();

        Tensor<T>& forward(Tensor<T>& input_tensor , bool Istraining = true) override;
        Tensor<T>& backpropagation(Tensor<T>& output_error, float learning_rate = 0.0001f) override;

    private:
        Tensor<T>& forward_2d(Tensor<T>& input_tensor, bool Istraining = true) ;
        Tensor<T>& forward_4d(Tensor<T>& input_tensor, bool Istraining = true) ;

        Tensor<T>& backpropagation_2d(Tensor<T>& output_error, float learning_rate = 0.0001f) ;
        Tensor<T>& backpropagation_4d(Tensor<T>& output_error, float learning_rate = 0.0001f) ;
    };



} 
