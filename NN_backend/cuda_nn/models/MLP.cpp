#include "MLP.h"
 
 
namespace Hex
{
	template <class T>
	MLP<T>::MLP(int input_size, int output_size, int batch_size,   int h_l_dimension) :
		 

		linear1(input_size, h_l_dimension, batch_size),
		relu1(),
		bn1(h_l_dimension, TensorShape::_2D),

		linear2(h_l_dimension, h_l_dimension, batch_size),
		relu2(),
		//bn2(h_l_dimension, TensorShape::_2D),

		linear3(h_l_dimension, h_l_dimension, batch_size ),
		relu3(),
		//bn3(h_l_dimension, TensorShape::_2D),
		

		linear4(h_l_dimension, output_size, batch_size),
		sigmoid1()
  
	{}

	template<class T>
	MLP<T>::~MLP()
	{    
 
	}
	template<class T>
	Tensor<T>& MLP<T>::forward(Tensor<T>& input_tensor, bool Istraining)
	{
		//std::cout << std::endl;


		//input_tensor.print();
		//std::cout << std::endl;
		X = linear1.forward(input_tensor , Istraining);
	 	
		//X.print();
		X = relu1.forward(X , Istraining);
		X = bn1.forward(X, Istraining);
		//////// hidden layer
		 
			X = linear2.forward(X , Istraining);
			//X.print();
			X = relu2.forward(X , Istraining);
			 
			//X.print();
		//
 
		
		X = linear4.forward(X , Istraining);
		//X.print();
		X = sigmoid1.forward(X , Istraining);
		// X.print();
		return X;
	}

	template<class T>
	Tensor<T>& MLP<T>::backpropagation(Tensor<T>& output_error, float learning_rate  ) { return output_error; }

	template<class T>
	void MLP<T>::backpropa(Tensor<T>& output_error, float learning_rate  )
	{	
	 
		// Calculate gradients for the output layer
		X= sigmoid1.backpropagation(output_error, learning_rate);
		//X.print();
		X = linear4.backpropagation(X, learning_rate);
		//X.print();
	
		//X.print();
 
		 
		//	
		 
			X = relu2.backpropagation(X, learning_rate);
			//X.print();
			X = linear2.backpropagation(X, learning_rate);
			//X.print();
		 
		X = bn1.backpropagation(X, learning_rate);
		//X.print();
		// Backpropagate through the first hidden layer
		
		X = relu1.backpropagation(X, learning_rate);
		//X.print();
	
		  linear1.backpropagation(X, learning_rate);
		//X.print();
		 
	}


	// Explicit instantiation of the template class for supported types
	template class MLP<float>;
	template class MLP<int>;
	template class MLP<double>;
}