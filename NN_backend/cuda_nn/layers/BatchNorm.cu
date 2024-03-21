
#include "BatchNorm.h"
#include "../utils/tensor_oprations.h"
#include <cassert>
namespace Hex {

    template <class T>
    BatchNorm<T>::BatchNorm(int features_or_channels, TensorShape tensorshape, float momentum, float eps)
        : momentum(momentum), eps(eps), _Tshape(tensorshape),
        gamma(tensorshape == TensorShape::_4D ? std::make_shared<Tensor<T>>(std::vector<int>{ 1, features_or_channels, 1, 1 }, false) : std::make_shared<Tensor<T>>(std::vector<int>{ 1, features_or_channels }, false)),
        beta(tensorshape == TensorShape::_4D ? std::make_shared<Tensor<T>>(std::vector<int>{ 1, features_or_channels, 1, 1 }, false) : std::make_shared<Tensor<T>>(std::vector<int>{ 1, features_or_channels }, false)),
        running_mean(tensorshape == TensorShape::_4D ? std::make_shared<Tensor<T>>(std::vector<int>{ 1, features_or_channels, 1, 1 }, false) : std::make_shared<Tensor<T>>(std::vector<int>{ 1, features_or_channels }, false)),
        running_var(tensorshape == TensorShape::_4D ? std::make_shared<Tensor<T>>(std::vector<int>{ 1, features_or_channels, 1, 1 }, false) : std::make_shared<Tensor<T>>(std::vector<int>{ 1, features_or_channels }, false)),
        input_mean(tensorshape == TensorShape::_4D ? Tensor<T>({ 1, features_or_channels, 1, 1 }) : Tensor<T>({ 1, features_or_channels })),
        input_var(tensorshape == TensorShape::_4D ? Tensor<T>({ 1, features_or_channels, 1, 1 }) : Tensor<T>({ 1, features_or_channels }))
    {
        initTensorToOneOnGPU(*gamma);
        initTensorToOneOnGPU(*running_var);
    }

    template <class T>
    BatchNorm<T>::~BatchNorm() {
        // Destructor implementation
       // delete x_normalized; 
    }

    template <class T>
    Tensor<T>& BatchNorm<T>::forward(Tensor<T>& input_tensor, bool Istraining) {
        size_t tensor_dimensions = input_tensor.getShape().size();
        //std::cout << " size from batch norm forward : " << tensor_dimensions << std::endl;
        if (tensor_dimensions == 4 && _Tshape == TensorShape::_4D) {
            //gamma->print();
            return forward_4d(input_tensor, Istraining);
        }
        else if (tensor_dimensions == 2 && _Tshape == TensorShape::_2D) {
            return forward_2d(input_tensor, Istraining);
        }
        assert(false && "Invalid tensor dimensions or shape");
    }

    template<class T>
    __global__ void batchnorm_forward_2d_kernel(const T* __restrict__ input_data,
        T* __restrict__ output_data,
        const T* __restrict__ gamma_data,
        const T* __restrict__ beta_data,
        T* __restrict__ running_mean,
        T* __restrict__ running_variance,
        T* __restrict__ x_normalized,
        T* __restrict__ input_mean,
        T* __restrict__ input_var,
        const int features,
        const int batch_size,
        const float momentum,
        const float eps,
        const bool Istraining) {

        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < batch_size && col < features) {
            int input_idx = row * features + col;

            if (Istraining) {
                // Calculate mean
                if (threadIdx.x == 0) {
                    T sum = 0;
                    for (int b = 0; b < batch_size; ++b) {
                        int data_idx = b * features + col;
                        sum += input_data[data_idx];

                    }
                    input_mean[col] = sum / (batch_size);

                    // 
                    T diff = 0;
                    T sum_squares = 0.0f;
                    for (int b = 0; b < batch_size; ++b) {
                        int data_idx = b * features + col;
                        diff = input_data[data_idx] - input_mean[col];
                        sum_squares += diff * diff;
                    }
                    input_var[col] = sum_squares / (batch_size);
                    // printf(" input_var[col] %f \n", input_var[col]);
                }
                __syncthreads();

                x_normalized[input_idx] = (input_data[input_idx] - input_mean[col]) * (static_cast<T>(1) / sqrtf(input_var[col] + eps));
                output_data[input_idx] = gamma_data[col] * x_normalized[input_idx] + beta_data[col];

                running_mean[col] = momentum * running_mean[col] + (1 - momentum) * input_mean[col];
                running_variance[col] = momentum * running_variance[col] + (1 - momentum) * input_var[col];

            }
            else {
                x_normalized[input_idx] = (input_data[input_idx] - running_mean[col]) / sqrtf(running_variance[col] + eps);
                output_data[input_idx] = gamma_data[col] * x_normalized[input_idx] + beta_data[col];
            }
        }
    }

    template<class T>
    Tensor<T>& BatchNorm<T>::forward_2d(Tensor<T>& input_tensor, bool Istraining)
    {

        input = std::make_shared<Tensor<T>>(input_tensor);
        //  std::cout << "input" << std::endl; 
          //input->print();
         // std::cout   << std::endl;
        output.reset(new Tensor<T>({ input_tensor.getShape() }));
        x_normalized.reset(new Tensor<T>({ input_tensor.getShape() }));

        int _batch_size = input->getShape()[0];
        int _fetures = input->getShape()[1];


        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((_batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (_fetures + threadsPerBlock.y - 1) / threadsPerBlock.y);
        // input->print();
        batchnorm_forward_2d_kernel << < numBlocks, threadsPerBlock >> > (input->getData(),
            output->getData(),
            gamma->getData(),
            beta->getData(),
            running_mean->getData(),
            running_var->getData(),
            x_normalized->getData(),
            input_mean.getData(),
            input_var.getData(),
            _fetures,
            _batch_size,
            momentum,
            eps,
            Istraining);
        cudaDeviceSynchronize();
        //input->print();
        // input_mean.print();
        //  input_var.print();
         //x_normalized.print();
        //std::cout << "output" << std::endl;
        //  output->print();
        //   std::cout   << std::endl;
         //running_mean->print();
         //running_var->print();
         //x_normalized->print();
        //gamma->print();
        return *output;
    }

    template<class T>
    __global__ void old_batchnorm_forward_4d_kernel(const T* __restrict__ input_data,
        T* __restrict__ output_data,
        const T* __restrict__ gamma_data,
        const T* __restrict__ beta_data,
        T* __restrict__ running_mean,
        T* __restrict__ running_variance,
        T* __restrict__  x_normalized,
        T* __restrict__  input_mean,
        T* __restrict__  input_var,
        const int batch_size,
        const int out_channels,
        const int input_width,
        const int input_height,
        const float momentum,
        const float eps,
        const bool Istraining) {

        int batch_channel_idx = blockIdx.x;
        int batch_idx = batch_channel_idx / out_channels;
        int channel_idx = batch_channel_idx % out_channels;
        int output_row = blockIdx.y * blockDim.y + threadIdx.x;
        int output_col = blockIdx.z * blockDim.z + threadIdx.y;

        if (batch_idx < batch_size && channel_idx < out_channels && output_row < input_width && output_col < input_height) {
            int input_idx = batch_idx * out_channels * input_height * input_width + channel_idx * input_height * input_width + output_row * input_width + output_col;
            if (Istraining) {

                // Compute mean along height and width for each channel
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    T sum = 0;

                    for (int b = 0; b < batch_size; ++b) {
                        for (int i = 0; i < input_width; ++i) {
                            for (int j = 0; j < input_height; ++j) {
                                int data_idx = b * out_channels * input_height * input_width + channel_idx * input_height * input_width + i * input_height + j;
                                sum += input_data[data_idx];

                            }
                        }
                    }
                    input_mean[channel_idx] = sum / (batch_size * input_height * input_width);

                    T diff = 0;
                    T sum_squares = 0.0f;
                    for (int b = 0; b < batch_size; ++b) {
                        for (int i = 0; i < input_width; ++i) {
                            for (int j = 0; j < input_height; ++j) {
                                int data_idx = b * out_channels * input_height * input_width + channel_idx * input_height * input_width + i * input_height + j;
                                diff = input_data[data_idx] - input_mean[channel_idx];
                                sum_squares += diff * diff;
                            }
                        }
                    }
                    input_var[channel_idx] = sum_squares / (batch_size * input_height * input_width);

                }

                __syncthreads();

                x_normalized[input_idx] = (input_data[input_idx] - input_mean[channel_idx]) / sqrtf(input_var[channel_idx] + eps);
                output_data[input_idx] = gamma_data[channel_idx] * x_normalized[input_idx] + beta_data[channel_idx];

                //self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
                //    self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

                running_mean[channel_idx] = momentum * running_mean[channel_idx] + (1 - momentum) * input_mean[channel_idx];
                running_variance[channel_idx] = momentum * running_variance[channel_idx] + (1 - momentum) * input_var[channel_idx];

            }
            else {
                x_normalized[input_idx] = (input_data[input_idx] - running_mean[channel_idx]) / sqrtf(running_variance[channel_idx] + eps);
                output_data[input_idx] = gamma_data[channel_idx] * x_normalized[input_idx] + beta_data[channel_idx];
            }

        }
    }
    template<class T>
    __global__ void batchnorm_forward_4d_kernel(const T* __restrict__ input_data,
        T* __restrict__ output_data,
        const T* __restrict__ gamma_data,
        const T* __restrict__ beta_data,
        T* __restrict__ running_mean,
        T* __restrict__ running_variance,
        T* __restrict__  x_normalized,
        T* __restrict__  input_mean,
        T* __restrict__  input_var,
        const int batch_size,
        const int out_channels,
        const int input_width,
        const int input_height,
        const float momentum,
        const float eps,
        const bool Istraining) {

        int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
        int idx_z = blockIdx.z * blockDim.z + threadIdx.z;
        __shared__ T smeandata[64];
        __shared__ T svardata[64];


        smeandata[threadIdx.z] = 0.0f;
        svardata[threadIdx.z] = 0.0f;
        if (idx_x < input_width && idx_y < input_height && idx_z < batch_size * out_channels) {
            int b = idx_z / out_channels;
            int c = idx_z % out_channels;
            int index = (b * out_channels + c) * input_width * input_height + idx_y * input_width + idx_x;

            if (Istraining) {

                atomicAdd(&smeandata[threadIdx.z], input_data[index]);
                input_mean[c] = (static_cast<T>(1) / (batch_size * input_height * input_width)) * smeandata[threadIdx.z];

                atomicAdd(&svardata[threadIdx.z], (input_data[index] - input_mean[c]) * (input_data[index] - input_mean[c]));
                input_var[c] = (static_cast<T>(1) / (batch_size * input_height * input_width)) * svardata[threadIdx.z];
                __syncthreads();


                x_normalized[index] = (input_data[index] - input_mean[c]) * (static_cast<T>(1) / sqrtf(input_var[c] + eps));
                output_data[index] = gamma_data[c] * x_normalized[index] + beta_data[c];


                running_mean[c] = momentum * running_mean[c] + (1 - momentum) * input_mean[c];
                running_variance[c] = momentum * running_variance[c] + (1 - momentum) * input_var[c];
            }
            else {
                x_normalized[index] = (input_data[index] - running_mean[c]) * (static_cast<T>(1) / sqrtf(running_variance[c] + eps));
                output_data[index] = gamma_data[c] * x_normalized[index] + beta_data[c];
            }
        }
    }

    template<class T>
    Tensor<T>& BatchNorm<T>::forward_4d(Tensor<T>& input_tensor, bool Istraining)
    {
        input = std::make_shared<Tensor<T>>(input_tensor);

        output.reset(new Tensor<T>({ input->getShape() }));
        x_normalized.reset(new Tensor<T>({ input->getShape() }));

        int  _batch_size = input->getShape()[0];
        int  _out_channels = input->getShape()[1];
        int  _in_width = input->getShape()[2];
        int  _in_height = input->getShape()[3];

        dim3 threadsPerBlock(8, 8, 8);
        dim3 numBlocks((_in_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (_in_height + threadsPerBlock.y - 1) / threadsPerBlock.y,
            (_batch_size * _out_channels + threadsPerBlock.z - 1) / threadsPerBlock.z);


        //// Compute mean and variance
        //Tensor<T> mean({ 1, _out_channels,1,1 });
        //Tensor<T> variance({ 1, _out_channels,1,1 });



        batchnorm_forward_4d_kernel << <numBlocks, threadsPerBlock >> > (input->getData(),
            output->getData(),
            gamma->getData(),
            beta->getData(),
            running_mean->getData(),
            running_var->getData(),
            x_normalized->getData(),
            input_mean.getData(),
            input_var.getData(),
            _batch_size,
            _out_channels,
            _in_width,
            _in_height,
            momentum,
            eps,
            Istraining);
        cudaDeviceSynchronize();
        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Error from batchnorm 4d forward method: %s\n", cudaGetErrorString(cudaError));
            exit(EXIT_FAILURE);  // or handle the error appropriately
        }
        // input->print();
         // input_mean.print();
         //input_var.print();
         //input_mean.print();
         //input_var.print();
         //x_normalized.print();
         //output->print();
         //running_mean->print();
         //running_var->print();
        return *output;
    }




    template <class T>
    Tensor<T>& BatchNorm<T>::backpropagation(Tensor<T>& output_error, float learning_rate) {

        size_t tensor_dimensions = output_error.getShape().size();
        //std::cout << " size from batch norm forward : " << tensor_dimensions << std::endl;
        if (tensor_dimensions == 4 && _Tshape == TensorShape::_4D) {
            //gamma->print();
            return backpropagation_4d(output_error, learning_rate);
        }
        else if (tensor_dimensions == 2 && _Tshape == TensorShape::_2D) {
            return backpropagation_2d(output_error, learning_rate);
        }
        assert(false && "Invalid tensor dimensions or shape");

    }


    template<class T>
    __global__ void batchnorm_backward_2d_kernel(const T* __restrict__ input_data,
        const T* __restrict__ output_error,
        const T* __restrict__ x_normalized,
        const T* __restrict__ input_mean,
        const T* __restrict__ input_var,
        T* __restrict__ input_error,
        T* __restrict__ gamma_gradient,
        T* __restrict__ beta_gradient,
        T* __restrict__ grad_normalized,
        const int features,
        const int batch_size,
        const float eps,
        const float learning_rate)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;
        const  T clip_threshold = static_cast<T>(0.1);
        if (row < batch_size && col < features) {
            int input_idx = row * features + col;
            T grad_gamma = 0.0;
            T grad_beta = 0.0;

            // Calculate gradient of beta and gamma


            grad_normalized[input_idx] = output_error[input_idx] * gamma_gradient[col];

            // Calculate dvar

            T dvar = 0.0f;
            if (threadIdx.x == 0) {

                for (int b = 0; b < batch_size; ++b) {
                    int data_idx = b * features + col;
                    T r = (input_data[data_idx] - input_mean[col]);
                    T t = pow(input_var[col] + eps, -1.5);
                    dvar += grad_normalized[data_idx] * r * -0.5 * t;

                }

            }
            __syncthreads();

            // Calculate dmean

            T dmean = 0.0f;
            if (threadIdx.x == 0) {

                T a = 0.0;
                T d = 0.0;
                for (int b = 0; b < batch_size; ++b) {
                    int data_idx = b * features + col;
                    a += grad_normalized[data_idx] * (-1 / sqrt(input_var[col] + eps));
                    d += (-2 * (input_data[data_idx] - input_mean[col])) / batch_size;
                }
                dmean = a * dvar + d;
            }
            __syncthreads();

            for (int b = 0; b < batch_size; ++b) {
                int data_idx = b * features + col;

                input_error[data_idx] = grad_normalized[data_idx] / sqrt(input_var[col] + eps) + dvar * 2.0 * (input_data[data_idx] - input_mean[col]) / batch_size + dmean / batch_size;
                if (input_error[data_idx] > clip_threshold) {
                    input_error[data_idx] = clip_threshold;
                }
                else if (input_error[data_idx] < -clip_threshold) {
                    input_error[data_idx] = -clip_threshold;
                }
            }

            if (threadIdx.x == 0) {


                for (int b = 0; b < batch_size; ++b) {
                    int data_idx = b * features + col;
                    grad_gamma += output_error[data_idx] * x_normalized[data_idx];
                    grad_beta += output_error[data_idx];
                }


                if (grad_gamma  > clip_threshold) {
                    grad_gamma  = clip_threshold;
                }
                else if (grad_gamma  < -clip_threshold) {
                    grad_gamma  = -clip_threshold;
                }

                if (grad_beta  > clip_threshold) {
                    grad_beta  = clip_threshold;
                }
                else if (grad_beta  < -clip_threshold) {
                    grad_beta  = -clip_threshold;
                }

                gamma_gradient[col] -= grad_gamma;
                beta_gradient[col] -= grad_beta;
            }
            __syncthreads();

        }
    }


    template<class T>
    Tensor<T>& BatchNorm<T>::backpropagation_2d(Tensor<T>& output_error, float learning_rate)
    {
        input_error.reset(new Tensor<T>({ input->getShape() }));
        grad_normalized.reset(new Tensor<T>({ input->getShape() }));
        const int batch_size = input->getShape()[0];
        const int features = input->getShape()[1];
        // gamma->print();
         // input->print();
        const dim3 blockSize(16, 16); // Adjust block size as needed
        const dim3 gridSize((batch_size + blockSize.x - 1) / blockSize.x, (features + blockSize.y - 1) / blockSize.y); // Adjust grid size as needed
        // x_normalized->print();
         // Invoke the CUDA kernel for backpropagation
        batchnorm_backward_2d_kernel << <gridSize, blockSize >> > (input->getData(),
            output_error.getData(),
            x_normalized->getData(),
            input_mean.getData(),
            input_var.getData(),
            input_error->getData(),
            gamma->getData(),
            beta->getData(),
            grad_normalized->getData(),
            features,
            batch_size,
            eps, learning_rate);

        // Synchronize to ensure the kernel is finished
        cudaDeviceSynchronize();
        //std::cout << " aaaaaaaaaaaaaaa" << std::endl;
        // gamma->print();

        // beta->print();
        //input_error->print();
        return *input_error;
    }


    template<class T>
    __global__ void old_batchnorm_backward_4d_kernel(const T* __restrict__ input_data,
        const T* __restrict__ output_error,
        const T* __restrict__ x_normalized,
        const T* __restrict__ input_mean,
        const T* __restrict__ input_var,
        T* __restrict__ input_error,
        T* __restrict__ gamma_gradient,
        T* __restrict__ beta_gradient,
        T* __restrict__ grad_normalized,
        const int batch_size,
        const int out_channels,
        const int input_width,
        const int input_height,
        const float eps)
    {
        int batch_channel_idx = blockIdx.x;
        int batch_idx = batch_channel_idx / out_channels;
        int channel_idx = batch_channel_idx % out_channels;
        int output_row = blockIdx.y * blockDim.y + threadIdx.x;
        int output_col = blockIdx.z * blockDim.z + threadIdx.y;

        if (batch_idx < batch_size && channel_idx < out_channels && output_row < input_width && output_col < input_height) {
        }

        int input_idx = batch_idx * out_channels * input_height * input_width + channel_idx * input_height * input_width + output_row * input_width + output_col;

        grad_normalized[input_idx] = output_error[input_idx] * gamma_gradient[channel_idx];

        T dvar = 0.0f;
        if (threadIdx.x == 0 && threadIdx.y == 0) {

            for (int b = 0; b < batch_size; ++b) {
                for (int i = 0; i < input_width; ++i) {
                    for (int j = 0; j < input_height; ++j) {
                        int data_idx = b * out_channels * input_height * input_width + channel_idx * input_height * input_width + i * input_height + j;
                        T r = (input_data[data_idx] - input_mean[channel_idx]);
                        T t = pow(input_var[channel_idx] + eps, -1.5);
                        dvar += grad_normalized[data_idx] * r * -0.5 * t;
                        // printf("dvar %f \n", dvar);
                    }
                }
            }

        }
        __syncthreads();

        T dmean = 0.0f;
        if (threadIdx.x == 0 && threadIdx.y == 0) {

            T a = 0.0;
            T d = 0.0;
            for (int b = 0; b < batch_size; ++b) {
                for (int i = 0; i < input_width; ++i) {
                    for (int j = 0; j < input_height; ++j) {
                        int data_idx = b * out_channels * input_height * input_width + channel_idx * input_height * input_width + i * input_height + j;
                        a += grad_normalized[data_idx] * (-1 / sqrt(input_var[channel_idx] + eps));
                        d += (-2 * (input_data[data_idx] - input_mean[channel_idx])) / (batch_size * input_height * input_width);

                    }
                }
            }
            dmean = a * dvar + d;
            //  printf("dvar %lf \n", dmean);
        }
        __syncthreads();

        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < input_width; ++i) {
                for (int j = 0; j < input_height; ++j) {
                    int data_idx = b * out_channels * input_height * input_width + channel_idx * input_height * input_width + i * input_height + j;
                    input_error[data_idx] = grad_normalized[data_idx] / sqrt(input_var[channel_idx] + eps) + dvar * 2.0 * (input_data[data_idx] - input_mean[channel_idx]) / (batch_size * input_height * input_width) + dmean / (batch_size * input_height * input_width);
                    // printf("dvar %f \n", dvar);
                }
            }
        }

        T grad_gamma = 0.0;
        T grad_beta = 0.0;
        if (threadIdx.x == 0 && threadIdx.y == 0) {

            for (int b = 0; b < batch_size; ++b) {
                for (int i = 0; i < input_width; ++i) {
                    for (int j = 0; j < input_height; ++j) {
                        int data_idx = b * out_channels * input_height * input_width + channel_idx * input_height * input_width + i * input_height + j;
                        grad_gamma += output_error[data_idx] * x_normalized[data_idx];
                        grad_beta += output_error[data_idx];
                    }
                }
            }

            gamma_gradient[channel_idx] -= grad_gamma;
            beta_gradient[channel_idx] -= grad_beta;
        }
        __syncthreads();

    }

    template<class T>
    __global__ void  batchnorm_backward_4d_kernel(const T* __restrict__ input_data,
        const T* __restrict__ output_error,
        const T* __restrict__ x_normalized,
        const T* __restrict__ input_mean,
        const T* __restrict__ input_var,
        T* __restrict__ input_error,
        T* __restrict__ gamma_gradient,
        T* __restrict__ beta_gradient,
        T* __restrict__ grad_normalized,
        const int batch_size,
        const int out_channels,
        const int input_width,
        const int input_height,
        const float eps)
    {
        const  T clip_threshold = static_cast<T>(0.1);
        int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
        int idx_z = blockIdx.z * blockDim.z + threadIdx.z;
        __shared__ T dmeandata[64];
        __shared__ T dvardata[64];

        __shared__ T a[64];
        __shared__ T k[64];
        __shared__ T grad_gamma[64];
        __shared__ T grad_beta[64];


        dmeandata[threadIdx.z] = 0.0f;
        dvardata[threadIdx.z] = 0.0f;
        a[threadIdx.z] = 0.0f;
        k[threadIdx.z] = 0.0f;
        grad_gamma[threadIdx.z] = 0.0f;
        grad_beta[threadIdx.z] = 0.0f;
        if (idx_x < input_width && idx_y < input_height && idx_z < batch_size * out_channels) {
            int b = idx_z / out_channels;
            int c = idx_z % out_channels;
            int index = (b * out_channels + c) * input_width * input_height + idx_y * input_width + idx_x;

            grad_normalized[index] = output_error[index] * gamma_gradient[c];

            T r = (input_data[index] - input_mean[c]);
            T t = pow(input_var[c] + eps, -1.5);
            atomicAdd(&dvardata[threadIdx.z], grad_normalized[index] * r * -0.5 * t);


            atomicAdd(&a[threadIdx.z], grad_normalized[index] * (-1 / sqrt(input_var[c] + eps)));
            atomicAdd(&k[threadIdx.z], -2 * ((static_cast<T>(1) / (batch_size * input_height * input_width)) * (input_data[index] - input_mean[c])));
            dmeandata[threadIdx.z] = a[threadIdx.z] * dvardata[threadIdx.z] + k[threadIdx.z];


            input_error[index] = grad_normalized[index] * (static_cast<T>(1) / sqrt(input_var[c] + eps)) + dvardata[threadIdx.z] * 2.0 *
                ((static_cast<T>(1) / (batch_size * input_height * input_width)) * (input_data[index] - input_mean[c])) +
                ((static_cast<T>(1) / (batch_size * input_height * input_width)) * dmeandata[threadIdx.z]);

            if (input_error[index] > clip_threshold) {
                input_error[index] = clip_threshold;
            }
            else if (input_error[index] < -clip_threshold) {
                input_error[index] = -clip_threshold;
            }

         

            atomicAdd(&grad_gamma[threadIdx.z], output_error[index] * x_normalized[index]);
            atomicAdd(&grad_beta[threadIdx.z], output_error[index]);

            if (grad_gamma[threadIdx.z] > clip_threshold) {
                grad_gamma[threadIdx.z] = clip_threshold;
            }
            else if (grad_gamma[threadIdx.z] < -clip_threshold) {
                grad_gamma[threadIdx.z] = -clip_threshold;
            }

            if (grad_beta[threadIdx.z] > clip_threshold) {
                grad_beta[threadIdx.z] = clip_threshold;
            }
            else if (grad_beta[threadIdx.z] < -clip_threshold) {
                grad_beta[threadIdx.z] = -clip_threshold;
            } 

            gamma_gradient[c] -= grad_gamma[threadIdx.z];
            beta_gradient[c] -= grad_beta[threadIdx.z];

            __syncthreads();
        }

    }


    template<class T>
    Tensor<T>& BatchNorm<T>::backpropagation_4d(Tensor<T>& output_error, float learning_rate)
    {
        input_error.reset(new Tensor<T>({ input->getShape() }));
        grad_normalized.reset(new Tensor<T>({ input->getShape() }));
        /*       const int _batch_size = input->getShape()[0];
               const int _out_channels = input->getShape()[1];
               const int _in_width = input->getShape()[2];
               const int _in_height = input->getShape()[3];

               dim3 threadsPerBlock(8, 8, 8);
               dim3 numBlocks(_batch_size * _out_channels,
                   (_in_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (_in_height + threadsPerBlock.y - 1) / threadsPerBlock.y);*/



        int  _batch_size = input->getShape()[0];
        int  _out_channels = input->getShape()[1];
        int  _in_width = input->getShape()[2];
        int  _in_height = input->getShape()[3];

        dim3 threadsPerBlock(8, 8, 8);
        dim3 numBlocks((_in_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (_in_height + threadsPerBlock.y - 1) / threadsPerBlock.y,
            (_batch_size * _out_channels + threadsPerBlock.z - 1) / threadsPerBlock.z);

        batchnorm_backward_4d_kernel << <numBlocks, threadsPerBlock >> > (input->getData(),
            output_error.getData(),
            x_normalized->getData(),
            input_mean.getData(),
            input_var.getData(),
            input_error->getData(),
            gamma->getData(),
            beta->getData(),
            grad_normalized->getData(),
            _batch_size,
            _out_channels,
            _in_width,
            _in_height,
            eps);
        cudaDeviceSynchronize();

        // gamma->print();
         // beta->print();
        return *input_error;
    }

    template class BatchNorm<float>;
    template class BatchNorm<int>;
    template class BatchNorm<double>;
}