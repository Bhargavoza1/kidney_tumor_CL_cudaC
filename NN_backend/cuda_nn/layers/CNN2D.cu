#include "CNN2D.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <memory>
//Output size = ((input size - kernel size + 2 * padding) / stride) + 1
//
// 
//
//input size = (Output size - 1) * stride + kernel size - 2 * padding

namespace Hex
{
    template<class T>
    __global__ void cnn2d_W_B_init(T* weights, T* bias, int out_channels, int in_channels, int kernel_size, float w_b_range) {
 

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        // Initialize random number generator for each thread

        for (int i = idx; i < out_channels * in_channels * kernel_size * kernel_size; i += stride) {
            int row = i / (in_channels * kernel_size * kernel_size);
            int col = i % (in_channels * kernel_size * kernel_size);

            curandState state;
            curand_init(777, idx, 0, &state);

            if (col == 0) {
                bias[row] = curand_uniform(&state) * (2 * w_b_range) - w_b_range;
                
            }
            weights[i] = curand_uniform(&state) * (2 * w_b_range) - w_b_range;
           

            /////// init weights and bias for test

            //if (col == 0) {
            //   
            //    bias[row] = static_cast<T>(row  );
            //}
            //
            //weights[i] = static_cast<T>(i);
        }
 
    }


    template<class T>
    void CNN2D<T>::init_weight_n_bias()
    {
        dim3 blockSize(256);
        dim3 gridSize((_out_channels * _in_channels * _kernel_size * _kernel_size + blockSize.x - 1) / blockSize.x);

        cnn2d_W_B_init << <gridSize, blockSize >> > (weights->getData(), bias->getData(), _out_channels, _in_channels, _kernel_size, _w_b_range );
        cudaDeviceSynchronize();
        //weights->print();
        //bias->print();
    }

    template<class T>
    CNN2D<T>::CNN2D(const int batch_size, const std::vector<int>& in_out_channels, int kernel_size, int padding,int stride,float w_b_range) :
        _batch_size(batch_size), _in_channels(in_out_channels[0]), _out_channels(in_out_channels[1]), _kernel_size(kernel_size),
        _padding(padding), _stride(stride), _w_b_range(w_b_range),
        weights(std::make_shared<Tensor<T>>(std::vector<int>{_out_channels, _in_channels, _kernel_size, _kernel_size  }   , false)),
        bias(std::make_shared<Tensor<T>>(std::vector<int>{_out_channels}, false))
       // output(std::vector<int>{_batch_size, _out_channels, batch_width_height[2], batch_width_height[3] }),
      //  input(std::vector<int>{_batch_size, _in_channels, batch_width_height[2], batch_width_height[3] }),
       // input_error(std::vector<int>{_batch_size, _in_channels, batch_width_height[2], batch_width_height[3]  })
    {
        init_weight_n_bias();
    }

    template<class T>
    CNN2D<T>::~CNN2D()
    {
        output->cudafree();
        input->cudafree();
        input_error->cudafree();
    }
 
  

    template<class T>
    __global__ void convolutionforward(const T* input, const T* weight, const T* bias, T* output,
        int batch_size, int in_channels, int in_width, int in_height,
        int out_channels, int kernel_size, int padding, int stride, int out_width, int out_height) {

        int batch_idx = blockIdx.x / out_channels;
        int channel_idx = blockIdx.x % out_channels;
        int output_row = blockIdx.y * blockDim.y + threadIdx.x;
        int output_col = blockIdx.z * blockDim.z + threadIdx.y;
      
        if (batch_idx < batch_size && channel_idx < out_channels && output_row < out_width && output_col < out_height) {
            int input_row_start = output_row * stride - padding;
            int input_col_start = output_col * stride - padding;
            T value = 0;

            for (int c = 0; c < in_channels; ++c) {
                for (int i = 0; i < kernel_size; ++i) {
                    for (int j = 0; j < kernel_size; ++j) {
                        int input_row = input_row_start + i;
                        int input_col = input_col_start + j;
                        if (input_row >= 0 && input_row < in_height && input_col >= 0 && input_col < in_width) {
                            int input_idx = (batch_idx * in_channels * in_height * in_width) + (c * in_height * in_width) + (input_row * in_width) + input_col;
                            int weight_idx = (channel_idx * in_channels * kernel_size * kernel_size) + (c * kernel_size * kernel_size) + (i * kernel_size) + j;
                          
                            value += input[input_idx] * weight[weight_idx];
                          //  printf("(%dx%dx%dx%d  \n", batch_idx , c, output_row, output_col);
                        }
                    }
                }
            }
           // printf("(%f  \n", value);
            int output_idx = (batch_idx * out_channels * out_height * out_width) + (channel_idx * out_height * out_width) + (output_row * out_width) + output_col;
            output[output_idx] = value  ; 
            //printf("(%dx%dx%dx%d)  \n", batch_idx ,  channel_idx  , output_row , output_col);
        }
    }

 

    template<class T>
    __global__ void new_convolutionforward(const T* input, const T* weight, const T* bias, T* output,
        int batch_size, int in_channels, int in_width, int in_height,
        int out_channels, int kernel_size, int padding, int stride, int out_width, int out_height) {

        int tx = blockIdx.x * blockDim.x + threadIdx.x;
        int ty = blockIdx.y * blockDim.y + threadIdx.y;
        int tz = blockIdx.z * blockDim.z + threadIdx.z;
        int b = tz / out_channels;
        int c = tz % out_channels;

        if (tz < batch_size * out_channels && tx < out_width && ty < out_height) {
            int input_row_start = tx * stride - padding;
            int input_col_start = ty * stride - padding;
            T value = 0;

         
                for (int i = 0; i < kernel_size; ++i) {
                    int input_row = input_row_start + i;
                    if (input_row >= 0 && input_row < in_width){
                        for (int j = 0; j < kernel_size; ++j) {
                            int input_col = input_col_start + j;
                            if (input_col >= 0 && input_col < in_height) {
                                for (int ic = 0; ic < in_channels; ++ic) {
                                int input_idx = (b * in_channels * in_height * in_width) + (ic * in_height * in_width) + (input_row * in_height) + input_col;
                                int weight_idx = (c * in_channels * kernel_size * kernel_size) + (ic * kernel_size * kernel_size) + (i * kernel_size) + j;
                                value += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                }

            int output_idx = (b * out_channels * out_height * out_width) + (c * out_height * out_width) + (tx * out_width) + ty;
            output[output_idx] = value + bias[c]; // Add bias term
        }
    }









    template<class T>
    Tensor<T>& CNN2D<T>::forward(Tensor<T>& input_tensor, bool Istraining)
    {
     
        input = std::make_shared<Tensor<T>>(input_tensor);

    
        int  _batch_size = input->getShape()[0];
        int  _in_width = input->getShape()[2];
        int  _in_height = input->getShape()[3];
      
        int _out_width = ((_in_width - _kernel_size + 2 * _padding) / _stride) + 1;
        int _out_height = ((_in_height - _kernel_size + 2 * _padding) / _stride) + 1;

        //std::cout << _in_width << _in_height << _out_width << _out_height;
        output.reset(new Tensor<T>({ _batch_size , _out_channels ,_out_width , _out_height }));

        //// old threads 
        //dim3 threadsPerBlock(8,8,8);
        //dim3 numBlocks(_batch_size * _out_channels ,
        //    (_in_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        //    (_in_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

        ////// new threads

        dim3 threadsPerBlock(8, 8, 8);
        dim3 numBlocks((_in_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (_in_height + threadsPerBlock.y - 1) / threadsPerBlock.y,
            (_batch_size * _out_channels + threadsPerBlock.z - 1) / threadsPerBlock.z);
 
        new_convolutionforward << <numBlocks, threadsPerBlock >> > (input_tensor.getData(),
            weights->getData(), bias->getData(), output->getData(),
            _batch_size, _in_channels, _in_width, _in_height,
            _out_channels, _kernel_size, _padding, _stride, _out_width , _out_height);
        cudaDeviceSynchronize();
        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("error from liner backword method : %s\n", cudaGetErrorString(cudaError));
            exit(EXIT_FAILURE);  // or handle the error appropriately
        }
 /*        std::cout << "weights" << std::endl;
          weights->print();
          bias->print();*/
      //  output->print();
        return *output; 
    }


    template<class T>
    __global__ void convolutionBackwardInputError(const T* output_error, const T* input, T* weight, T* bias, T* input_error,
        int batch_size, int in_channels, int in_width, int in_height,
        int out_channels, int kernel_size, int padding, int stride, int out_width, int out_height , float learning_rate) {

        int batch_idx = blockIdx.x / in_channels;
        int channel_idx = blockIdx.x % in_channels;
        int input_row = blockIdx.y * blockDim.y + threadIdx.x;
        int input_col = blockIdx.z * blockDim.z + threadIdx.y;

        if (batch_idx < batch_size && channel_idx < in_channels && input_row < in_height && input_col < in_width) {
            T value = 0;
           
            int input_idx = (batch_idx * in_channels * in_height * in_width) + (channel_idx * in_height * in_width) + (input_row * in_width) + input_col;
            for (int c = 0; c < out_channels; ++c) {
                for (int i = 0; i < kernel_size; ++i) {
                    for (int j = 0; j < kernel_size; ++j) {
                        int output_row = (input_row + padding - i) / stride;
                        int output_col = (input_col + padding - j) / stride;
                        if (output_row >= 0 && output_row < out_height && output_col >= 0 && output_col < out_width) {
                            int output_idx = (batch_idx * out_channels * out_height * out_width) + (c * out_height * out_width) + (output_row * out_width) + output_col;
                            int weight_idx = (c * in_channels * kernel_size * kernel_size) + (channel_idx * kernel_size * kernel_size) + (i * kernel_size) + j;
                            value += output_error[output_idx] * weight[weight_idx];

                            printf("weight[weight_idx] %d \n", weight_idx);
                            ////////////////////// Update weight only if within bounds
                            // printf("before weight[weight_idx] %d = %f \n", weight_idx, weight[weight_idx]);
                            //if (input_row - padding + i * stride >= 0 && input_col - padding + j * stride >= 0) {
                            atomicAdd(&weight[weight_idx], -learning_rate * output_error[output_idx] * input[input_idx]);
                            // }
                             // printf("after weight[weight_idx] %d = %f \n", weight_idx , weight[weight_idx]);
                            // No need to update bias here since it's handled outside the loop
                        }
                    }
                }
            }
           // int input_idx = (batch_idx * in_channels * in_height * in_width) + (channel_idx * in_height * in_width) + (input_row * in_width) + input_col;
            input_error[input_idx] = value;
        }
 
        // Update bias
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            for (int c = 0; c < out_channels; ++c) { 
                 //printf("before bias[c] %d =  %f \n",c , bias[c]);
                atomicAdd(&bias[c], -learning_rate * output_error[(batch_idx * out_channels * out_height * out_width) + (c * out_height * out_width)]);
                 // printf("after bias[c]  %d =  %f \n", c, bias[c]);
            }
        }
    }
    
    
 
    template<class T>
    __global__ void new_convolutionBackwardInputError(const T* output_error, const T* input, T* weight, T* bias, T* input_error,
        int batch_size, int in_channels, int in_width, int in_height,
        int out_channels, int kernel_size, int padding, int stride, int out_width, int out_height, float learning_rate) {

        int tx = blockIdx.x * blockDim.x + threadIdx.x;
        int ty = blockIdx.y * blockDim.y + threadIdx.y;
        int tz = blockIdx.z * blockDim.z + threadIdx.z;


        if (tx < in_width && ty < in_height && tz < batch_size * in_channels) {
            int b = tz / in_channels;
            int channel_idx = tz % in_channels;
           
            int input_idx = (b * in_channels * in_height * in_width) + (channel_idx * in_height * in_width) + (tx * in_height) + ty;

            for (int i = 0; i < kernel_size; ++i) {
                int output_row = (tx + padding - i) / stride;
                if (output_row >= 0 && output_row < out_height){
                    for (int j = 0; j < kernel_size; ++j) {
                        int output_col = (ty + padding - j) / stride;
                        if (output_col >= 0 && output_col < out_width) {
                            for (int c = 0; c < out_channels; ++c) {
                             
                                    int output_idx = (b * out_channels * out_height * out_width) + (c * out_height * out_width) + (output_row * out_width) + output_col;
                                    int weight_idx = (c * in_channels * kernel_size * kernel_size) + (channel_idx * kernel_size * kernel_size) + (i * kernel_size) + j;
                                    input_error[input_idx] += output_error[output_idx] * weight[weight_idx];

                                    weight[weight_idx] -= learning_rate * output_error[output_idx] * input[input_idx];
                                    bias[c] -= learning_rate * output_error[output_idx];
                            }
                        }
                    }
                }
            }
        }
    }


    template<class T>
    Tensor<T>& CNN2D<T>::backpropagation(Tensor<T>& output_error, float learning_rate)
    {
        int  _batch_size = input->getShape()[0];
       // int  _in_width = input->getShape()[2];
       // int  _in_height = input->getShape()[3];
        int _out_width = output_error.getShape()[2];
        int _out_height = output_error.getShape()[3];
        int _in_width = (_out_width - 1) * _stride - 2 * _padding + _kernel_size;
        int _in_height = (_out_height - 1) * _stride - 2 * _padding + _kernel_size; 
       // output_error.print();

        input_error.reset(new Tensor<T>({ _batch_size , _in_channels ,_in_width , _in_height }));
       // bias->print();
        //dim3 threadsPerBlock(8 ,8, 8);
        //dim3 numBlocks(_batch_size * _in_channels,
        //    (_in_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        //    (_in_height + threadsPerBlock.y - 1) / threadsPerBlock.y
        //);
 
       /////// new kernel setting
        dim3 threadsPerBlock(8, 8, 8);
        dim3 numBlocks((_in_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (_in_height + threadsPerBlock.y - 1) / threadsPerBlock.y,
            (_batch_size * _in_channels + threadsPerBlock.z - 1) / threadsPerBlock.z);


       new_convolutionBackwardInputError << <numBlocks, threadsPerBlock >> > ( output_error.getData(), input->getData() ,
            weights->getData(), bias->getData() , input_error->getData(), _batch_size, _in_channels, _in_width, _in_height,
            _out_channels, _kernel_size, _padding, _stride, _out_width, _out_height, learning_rate);

        cudaDeviceSynchronize();
        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("error from liner backword method : %s\n", cudaGetErrorString(cudaError));
            exit(EXIT_FAILURE);  // or handle the error appropriately
        }
       // bias->print();
      //  weights->print();
        return *input_error;
    }




    template class CNN2D<float>;
    template class CNN2D<int>;
    template class CNN2D<double>;
}