#include "linear.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>
#include<iostream>
#include "../utils/Errorhelper.cpp"
namespace Hex {



	template<class T>
	linear<T>::linear(int input_size, int output_size, int batch_size, bool bias_as_zero, float w_b_range)
		: _bias_as_zero(bias_as_zero), _w_b_range(w_b_range), _batch_size(batch_size), _output_size(output_size),
		weights(std::make_shared<Tensor<T>>(std::vector<int>{ input_size, output_size } ,false)),
		bias(std::make_shared<Tensor<T>>(std::vector<int>{ 1, output_size }, false))
	{
		init_weight_n_bias();

	}


	template<class T>
	linear<T>::~linear()
	{
		output->cudafree();
		input->cudafree();
		input_error->cudafree();
	}

	template<class T>
	__global__ void linearLayerForward(const T* W, const T* X, T* Y, const T* b,
		int W_x_dim, int W_y_dim,
		int X_x_dim, int X_y_dim) {

		int row = blockIdx.x * blockDim.x + threadIdx.x;
		int col = blockIdx.y * blockDim.y + threadIdx.y;

		int Y_x_dim = X_x_dim;
		int Y_y_dim = W_y_dim;
		//printf("%d x %d", Y_x_dim, Y_y_dim);
		T Y_value = 0;

		if (row < Y_x_dim && col < Y_y_dim) {
			// Perform the matrix multiplication: Y = X * W  
			for (int i = 0; i < X_y_dim; ++i) {
				Y_value += X[row * X_y_dim + i] * W[i * W_y_dim + col];
			}

			// Add bias Y_value + b
			Y[row * Y_y_dim + col] = Y_value + b[col];
		}
	}

	template<class T>
	Tensor<T>& linear<T>::forward(Tensor<T>& input_tensor, bool Istraining)
	{
		
	
		input = std::make_shared<Tensor<T>>(input_tensor);
		output.reset(new Tensor<T>({ input_tensor.getShape()[0] , _output_size}));
	/*	std::cout << "input from forward" << std::endl;
		input->print();
		std::cout << std::endl;*/
		//if (weights.getShape()[0] != input->getShape()[1]) {
		//	std::cerr << "Error: Tensor shapes must be compatible for matrix multiplication. Shape of weights: "
		//		<< weights.getShape()[0] << "x" << weights.getShape()[1]
		//		<< ", Shape of input: " << input->getShape()[0] << "x" << input->getShape()[1] << std::endl;
		//	throw std::runtime_error("Tensor shape mismatch");
		//}


		//std::cout << "dbug strat of linear" << std::endl;

		//std::cout << "intpu" << std::endl;
		//input->print();
		//std::cout << std::endl;

		//std::cout << "weight" << std::endl;
		//weights.print();
		//std::cout << std::endl;
		//std::cout << "bias" << std::endl;
		//bias.print();
		//std::cout << std::endl;
		dim3 threadsPerBlock(16, 16);
		dim3 numBlocks((output->getShape()[0] + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(output->getShape()[1] + threadsPerBlock.y - 1) / threadsPerBlock.y);

		// Launch the forward kernel
		linearLayerForward << <numBlocks, threadsPerBlock >> > (weights->getData(), input->getData(), output->getData(), bias->getData(),
			weights->getShape()[0], weights->getShape()[1],
			input->getShape()[0], input->getShape()[1]);
		cudaDeviceSynchronize();

		//std::cout << "output" << std::endl;
		//output->print(); 
		//std::cout << std::endl;

		cudaError_t cudaError = cudaGetLastError();
		if (cudaError != cudaSuccess) {
			printf("Error from linear forward method: %s\n", cudaGetErrorString(cudaError));
			exit(EXIT_FAILURE);  // or handle the error appropriately
		}

		return *output;
	}

 

	template<typename T>
	__global__ void linear_backprop_kernel(T* input_error_data, const T* output_error_data, const T* weights_data, int batch_size, int input_size, int output_size) {
		int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
		int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
		// Assuming input_error_data, output_error_data, weights_data are row-major

		if (tid_x < input_size && tid_y < batch_size) {
			int idx = tid_y * input_size + tid_x;
			T sum = 0;
			for (int i = 0; i < output_size; ++i) {
				sum += output_error_data[tid_y * output_size + i] * weights_data[tid_x * output_size + i];
			}
			input_error_data[idx] = sum;
		}
	}

	template<typename T>
	__global__ void linear_update_weights_and_bias_kernel(T* weights, T* bias, const T* input, const T* output_error,
		int batch_size, int input_size, int output_size, float learning_rate) {

		int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
		int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

		if (idx_y < output_size) {
			if (idx_x < input_size) {
				// Update weights
				T weight_sum = 0;
				for (int i = 0; i < batch_size; ++i) {
					weight_sum += output_error[i * output_size + idx_y] * input[i * input_size + idx_x];
				}
				// printf("weight_sum %f : weight at %f  after lerning rate %f \n",  weight_sum , weights[idx_x * output_size + idx_y],weights[idx_x * output_size + idx_y] - learning_rate * weight_sum);
				weights[idx_x * output_size + idx_y] -= learning_rate * weight_sum;
			}
			// Update bias
			T bias_sum = 0;
			for (int i = 0; i < batch_size; ++i) {
				bias_sum += output_error[i * output_size + idx_y];
			}
			// printf("weight_sum %f\n", bias_sum);
			bias[idx_y] -= learning_rate * bias_sum;
		}
	}

	template<class T>
	Tensor<T>& linear<T>::backpropagation(Tensor<T>& output_error, float learning_rate)
	{
		//std::cout << std::endl;
		// std::cout << "output_error after update" << std::endl;
		//  output_error.print();

		input_error.reset(new Tensor<T>({ output_error.getShape()[0] ,input->getShape()[1] }));
		dim3 block_size(16, 16); // Adjust block size according to your GPU architecture
		dim3 grid_size((input->getShape()[1] + block_size.x - 1) / block_size.x,
			(input->getShape()[0] + block_size.y - 1) / block_size.y);

		linear_backprop_kernel << <grid_size, block_size >> > (input_error->getData(), output_error.getData(),
			weights->getData(), input->getShape()[0], input->getShape()[1], output_error.getShape()[1]);

		cudaDeviceSynchronize();

 

		dim3 blockDim(16, 16);
		dim3 gridDim((weights->getShape()[0] + blockDim.x - 1) / blockDim.x, (weights->getShape()[1] + blockDim.y - 1) / blockDim.y);
		linear_update_weights_and_bias_kernel << <gridDim, blockDim >> > (weights->getData(), bias->getData(),
			input->getData(), output_error.getData(), _batch_size, weights->getShape()[0], weights->getShape()[1], learning_rate);

		cudaDeviceSynchronize();
		cudaError_t cudaError = cudaGetLastError();
		if (cudaError != cudaSuccess) {
			printf("Error from linear backward method: %s\n", cudaGetErrorString(cudaError));
			exit(EXIT_FAILURE);  // or handle the error appropriately
		}



		//std::cout << std::endl;
		//std::cout << "bias after update" << std::endl;
		// bias.print();
		// std::cout << std::endl;
		// std::cout << "weights after update" << std::endl;
		//weights.print();
		//std::cout << std::endl;
		//std::cout << "input_error after update" << std::endl;
		//input_error->print();
		//std::cout << "dbug end of linear" << std::endl;
		//std::cout << std::endl;
		//std::cout << "input from backword" << std::endl;
	
		//input->print();
		//std::cout << std::endl;
		return *input_error;
	}



	template<class T>
	__global__ void initWeightKernel(T* weights, T* bias, int  input_size, int  output_size, int batch_size, bool bias_as_zero, float w_b_range) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;

		if (i < input_size && j < output_size) {
			// Random initialization of weights within the specified range
			curandState state;
			curand_init(777, i * output_size + j, 0, &state);

			float float_weight = (2 * curand_uniform(&state) - 1) * w_b_range;
			weights[i * output_size + j] = static_cast<T>(float_weight);



		}


		if (i == 0 && j < output_size) {

			if (bias_as_zero) {
				bias[j] = static_cast<T>(0.0);
			}
			else {
				curandState state_bias;
				curand_init(777, j, 0, &state_bias);

				float float_bias = (2 * curand_uniform(&state_bias) - 1) * w_b_range;
				bias[j] = static_cast<T>(float_bias);
			}
		}

		// int i = blockIdx.x * blockDim.x + threadIdx.x;
		//int j = blockIdx.y * blockDim.y + threadIdx.y;

		//int Y_x_dim = batch_size;
		//int Y_y_dim = output_size;

		//if (i < input_size && j < output_size) {
		//	 
		//	weights[i * output_size + j] = static_cast<T>(i * output_size + j + 1);
		//	
		//}

		//// Initialize bias if Isbias is true
		//if (  i == 0 && j <= Y_y_dim) {
		//	
		//	if (bias_as_zero) {
		//		bias[j] = static_cast<T>( (j) * 0.0);
		//	}
		//	else {
		//		 
		//		bias[j] = static_cast<T>((j)+ 1);
		//	}
		//} 
	}

	template<class T>
	void linear<T>::init_weight_n_bias() {
		dim3 threadsPerBlock(16, 16);
		dim3 numBlocks((weights->getShape()[0] + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(weights->getShape()[1] + threadsPerBlock.y - 1) / threadsPerBlock.y);

		// Launch the kernel to initialize weights and bias
		initWeightKernel << <numBlocks, threadsPerBlock >> > (weights->getData(), bias->getData(), weights->getShape()[0],
			weights->getShape()[1], _batch_size, _bias_as_zero, _w_b_range);
		cudaDeviceSynchronize();
	}




	template<class T>
	Tensor<T>& linear<T>::printW()
	{
		return *weights;
	}

	template<class T>
	Tensor<T>& linear<T>::printB()
	{
		return *bias;
	}

	// Explicit instantiation of the template class for supported types
	template class linear<float>;
	template class linear<int>;
	template class linear<double>;
}