#pragma once
#include "layer.h"
#include <iostream>
#include <memory>
namespace Hex {
	template <class T>
	class MaxPool2d : public layer<T>
	{
	private:
		int _kernel_size;
		int _padding;
		int _stride;

		std::shared_ptr<Tensor<T>> input;

		std::unique_ptr<Tensor<T>> output;
		std::unique_ptr<Tensor<T>> input_error;

	public:
		MaxPool2d(int kernel_size, int stride = 1, int padding = 0);
		~MaxPool2d();


		// Override forward method
		Tensor<T>& forward(Tensor<T>& input_tensor, bool Istraining = true) override;

		// Override backpropagation method
		Tensor<T>& backpropagation(Tensor<T>& output_error, float learning_rate = 0.001f) override;
	private:

	};

}