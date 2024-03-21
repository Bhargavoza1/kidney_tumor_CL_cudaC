#pragma once
#pragma once
#include "layer.h"

namespace Hex {

    template <class T>
    class flatten_layer : public layer<T> {
    private:
        int batch_size;
        int input_channels;
        int input_width;
        int input_height;
        int flattened_size;

    public:
        flatten_layer();

        ~flatten_layer() {}

        // Implementing forward method to flatten the input tensor
        Tensor<T>& forward(Tensor<T>& input_tensor, bool is_training = true) override;

        // Implementing backpropagation method
        Tensor<T>& backpropagation(Tensor<T>& output_error, float learning_rate = 0.0001f) override;
    };
}
