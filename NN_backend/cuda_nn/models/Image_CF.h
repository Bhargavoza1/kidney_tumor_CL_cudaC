#pragma once
#include "../layers/layer.h" 
#include "../layers/linear.h"
#include "../layers/ReLU.h"
#include "../layers/Sigmoid.h"
#include "../layers/BatchNorm.h"
#include "../layers/CNN2D.h"
#include "../layers/flatten_layer.h"
#include "../layers/MaxPool2d.h"
#include "../utils/Tensor.h"

namespace Hex {
    template <class T>
    class Image_CF : public layer<T>
    {
    private:

        
        Tensor<T> x;

        CNN2D<T>  conv1;
        ReLU<T>  relu1;
       // BatchNorm<T> bn1;
        MaxPool2d<T> pool1;

        CNN2D<T> conv2;
        ReLU<T>  relu2;
        BatchNorm<T> bn2;
        MaxPool2d<T> pool2;

        flatten_layer<T> fl;


        linear<T>  linear1;
        ReLU<T>  relu3;
        BatchNorm<T> bn3;
       
        linear<T>  linear2;
        Sigmoid<T>  sigmoid1;

    public:
        // Constructor
        Image_CF(int batch_size = 1, int input_channels = 3 , int output_class = 2 );
        ~Image_CF();

        // Override forward method
        Tensor<T>& forward(Tensor<T>& input_tensor, bool Istraining = true) override;

        // Override backpropagation method
        Tensor<T>& backpropagation(Tensor<T>& output_error, float learning_rate = 0.001f) override;

        void backpropa(Tensor<T>& output_error, float learning_rate = 0.001f);
    };

}


