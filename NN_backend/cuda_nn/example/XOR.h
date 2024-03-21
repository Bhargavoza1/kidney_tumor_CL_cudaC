#pragma once
#ifndef XOR_H
#define XOR_H

 
#include"../utils/Tensor.h"
#include "../utils/tensor_oprations.h"
#include "../models/MLP.h"
#include "../costs/MSE.h"

namespace Hex{

	template<typename T>
	void predictAndPrintResults(MLP<T>& model, const Tensor<T>& input_data, const Tensor<T>& target_data);

	template<typename T>
	void trainNeuralNetwork(MLP<T>& model, const Tensor<T>& input_data, const Tensor<T>& target_data, int num_epochs, T learning_rate);

	void xor_example();
}

 
#endif 
