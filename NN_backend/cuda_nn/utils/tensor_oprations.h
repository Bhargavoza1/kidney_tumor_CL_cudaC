#pragma once

#ifndef TENSOROP_H
#define TENSOROP_H

 #include <cuda_runtime.h> 
#include "Tensor.h"
#include <memory>

  namespace Hex{

	template<class T, class U>
	std::unique_ptr<Tensor<typename std::common_type<T, U>::type>> addTensor(const Tensor<T>& tensor1, const Tensor<U>& tensor2);

	template <typename T>
	void initTensorOnGPU(Tensor<T>& tensor, float multiplier);

	template <typename T>
	void initTensorToOneOnGPU(Tensor<T>& tensor );

	//template <typename T>
	//std::unique_ptr<Tensor<T>> slice(int index, Tensor<T> tensor);

	template <typename T>
	std::unique_ptr<Tensor<T>> transpose(const Tensor<T>& tensor);

	template <typename T>
	std::unique_ptr<Tensor<T>>  sliceFirstIndex(int firstIndex, const Tensor<T>& tensor);


 }

 
#endif  // TENSOROP_H