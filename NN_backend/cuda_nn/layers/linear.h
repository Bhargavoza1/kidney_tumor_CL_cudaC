#pragma once
#include "layer.h"
#include <iostream>
#include <memory>
namespace Hex {
    template<class T>
    class linear : public layer<T>
    {
    private:
        bool _Isbias;
        bool _bias_as_zero;
        float _w_b_range;
        int _batch_size;
        int _output_size;
        std::shared_ptr<Tensor<T>> weights;
        std::shared_ptr<Tensor<T>> bias;
        std::shared_ptr<Tensor<T>> input;


       

        std::unique_ptr<Tensor<T>> input_error;

        std::unique_ptr<Tensor<T>> output;

    public:
        // Constructor
       // linear() {};
        linear(int input_size, int output_size, int batch_size = 1, bool bias_as_zero = false , float w_b_range = 0.5f );
        ~linear();

        // Override forward method
        Tensor<T>& forward(Tensor<T>& input_tensor, bool Istraining = true) override;

        // Override backpropagation method
        Tensor<T>& backpropagation(Tensor<T>& output_error, float learning_rate = 0.001f) override;

        Tensor<T>& printW();
        Tensor<T>& printB();
        //Tensor<T>& printO();


    private:
        void init_weight_n_bias();
   
    };
}

// Include the implementation in a separate .cpp file if necessary
 