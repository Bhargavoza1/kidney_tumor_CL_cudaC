#include "flatten_layer.h"

namespace Hex {

    template<class T>
    flatten_layer<T>::flatten_layer()
    { 
    }
    // Implementing forward method to flatten the input tensor
        template <class T>
        Tensor<T>& flatten_layer<T>::forward(Tensor<T>& input_tensor, bool is_training ) {
            batch_size = input_tensor.getShape()[0]; // Batch dimension
            input_channels = input_tensor.getShape()[1];
            input_width = input_tensor.getShape()[2];
            input_height = input_tensor.getShape()[3];
            flattened_size = input_channels * input_width * input_height;

            // Reshape the tensor to a flat vector while preserving the batch dimension
            input_tensor.reshape({ batch_size, flattened_size });

            return input_tensor;
        }

        // Implementing backpropagation method
        template <class T>
        Tensor<T>& flatten_layer<T>::backpropagation(Tensor<T>& output_error, float learning_rate ) {
            output_error.reshape({ batch_size, input_channels, input_width, input_height });

            return output_error;
        }
 

        template class flatten_layer<float>;
        template class flatten_layer<int>;
        template class flatten_layer<double>;
}