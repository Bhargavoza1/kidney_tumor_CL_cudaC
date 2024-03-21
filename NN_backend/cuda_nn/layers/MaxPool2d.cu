
#include"MaxPool2d.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <limits>
using namespace std;
namespace Hex {
	template<class T>
	MaxPool2d<T>::MaxPool2d(int kernel_size, int stride, int padding):
		_kernel_size(kernel_size) , _padding(padding) , _stride(stride) 
	{
	}

	template<class T>
	MaxPool2d<T>::~MaxPool2d()
	{
	 
	}

 
	template <typename T>
	__global__ void maxpool2d_forward_kernel(const T* input, T* output,
		int batch_size, int channel, int input_width, int input_height,
		int output_width, int output_height,
		int kernel_size, int stride, int padding) {

		int batch_idx = blockIdx.x / channel;
		int channel_idx = blockIdx.x % channel;
		int output_row = blockIdx.y * blockDim.y + threadIdx.x;
		int output_col = blockIdx.z * blockDim.z + threadIdx.y;

		if (batch_idx < batch_size && channel_idx < channel  && output_row < output_width && output_col < output_height) {
			// Determine the starting point (top-left corner) of the corresponding input region
			int start_x = output_col * stride - padding;
			int start_y = output_row * stride - padding;

			// Find the limits of the region within the input
			int end_x =  min(start_x + kernel_size, input_width);
			int end_y =  min(start_y + kernel_size, input_height);

			start_x =  max(start_x, 0);
			start_y =  max(start_y, 0);

		 
			T max_val = static_cast<T>(0);

			// Iterate over the input region and find the maximum value
			for (int y = start_y; y < end_y; ++y) {
				for (int x = start_x; x < end_x; ++x) {
					T val = input[((batch_idx * channel + channel_idx) * input_height + y) * input_width + x];
					max_val = max(max_val, val);
				}
			}

			int output_idx = (batch_idx * channel * output_width * output_height) + (channel_idx * output_width * output_height) + (output_row * output_height) + output_col;
			// Set the output value to the maximum found
			output[output_idx] = max_val;
			// printf("output[output_idx] %f\n", channel_idx);
		}
	}


	template<class T>
	Tensor<T>& MaxPool2d<T>::forward(Tensor<T>& input_tensor, bool Istraining)
	{
		input = std::make_shared<Tensor<T>>(input_tensor);


		int  _batch_size = input->getShape()[0];
		int  _channel_size = input->getShape()[1];
		int  _in_width = input->getShape()[2];
		int  _in_height = input->getShape()[3]; 

		int _out_width = ((_in_width - _kernel_size + 2 * _padding) / _stride) + 1;
		int _out_height = ((_in_height - _kernel_size + 2 * _padding) / _stride) + 1;

		//std::cout << _in_width << _in_height << _out_width << _out_height;
		// in pooling we are only changging _out_width and _out_height other value will stay same 
		output.reset(new Tensor<T>({ _batch_size , _channel_size ,_out_width , _out_height }));

		dim3 threadsPerBlock(8, 8, 8);
		dim3 numBlocks(_batch_size * _channel_size,
			(_in_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(_in_height + threadsPerBlock.y - 1) / threadsPerBlock.y);  

		maxpool2d_forward_kernel << <numBlocks, threadsPerBlock >> > (input->getData(), output->getData(),
			_batch_size , _channel_size , _in_width, _in_height,
			_out_width, _out_height,
			_kernel_size, _stride, _padding);

		return *output;
	}



	template <typename T>
	__global__ void maxpool2d_backward_kernel(const T* input, const T* output_grad, T* input_grad,
		int batch_size, int channel, int output_width, int output_height,
		int input_width, int input_height,
		int kernel_size, int stride, int padding) {

		int batch_idx = blockIdx.x / channel;
		int channel_idx = blockIdx.x % channel;
		int output_row = blockIdx.y * blockDim.y + threadIdx.x;
		int output_col = blockIdx.z * blockDim.z + threadIdx.y;

		if (batch_idx < batch_size && channel_idx < channel && output_row < output_width && output_col < output_height) {
			// Determine the starting point (top-left corner) of the corresponding input region
			int start_x = output_col * stride - padding;
			int start_y = output_row * stride - padding;

			// Find the limits of the region within the input
			int end_x = min(start_x + kernel_size, input_width);
			int end_y = min(start_y + kernel_size, input_height);

			start_x = max(start_x, 0);
			start_y = max(start_y, 0);

			// Compute the gradient of the output with respect to the input
			T max_val = static_cast<T>(0);
			int max_x = start_x;
			int max_y = start_y;

			for (int y = start_y; y < end_y; ++y) {
				for (int x = start_x; x < end_x; ++x) {
					int input_idx = ((batch_idx * channel + channel_idx) * input_height + y) * input_width + x;
					T val = input[input_idx];
					if (val > max_val) {
						max_val = val;
						max_x = x;
						max_y = y;
					}
				}
			}

			int input_idx = ((batch_idx * channel + channel_idx) * input_height + max_y) * input_width + max_x; 
			int output_idx = (batch_idx * channel * output_width * output_height) + (channel_idx * output_width * output_height) + (output_row * output_height) + output_col;
			input_grad[input_idx] = output_grad[output_idx];
		}
	}

	template<class T>
	Tensor<T>& MaxPool2d<T>::backpropagation(Tensor<T>& output_error, float learning_rate)
	{

		int  _batch_size = output_error.getShape()[0];
		int  _channel_size = output_error.getShape()[1];
		int  _out_width = output_error.getShape()[2];
		int  _out_height = output_error.getShape()[3];


		int _in_width = (_out_width - 1) * _stride - 2 * _padding + _kernel_size;
		int _in_height = (_out_height - 1) * _stride - 2 * _padding + _kernel_size;

		input_error.reset(new Tensor<T>({ _batch_size , _channel_size ,_in_width , _in_height }));

		dim3 threadsPerBlock(8, 8, 8);
		dim3 numBlocks(_batch_size * _channel_size,
			(_in_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(_in_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

		maxpool2d_backward_kernel << <numBlocks, threadsPerBlock >> > (input->getData(), output_error.getData(), input_error->getData(),
			_batch_size, _channel_size, _out_width, _out_height,
			_in_width, _in_height,
			_kernel_size, _stride, _padding);

		return *input_error;
	 
	}

	template class MaxPool2d<float>;
	template class MaxPool2d<int>;
	template class MaxPool2d<double>;
}