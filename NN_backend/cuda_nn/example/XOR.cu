#include <iostream>
#include"../utils/Tensor.h"
#include "../utils/tensor_oprations.h"
#include "../models/MLP.h"
#include "../costs/MSE.h"

 

namespace Hex {

 template<typename T>
    void predictAndPrintResults(MLP<T>& model, const Tensor<T>& input_data, const Tensor<T>& target_data) {
        std::vector<int> input_shape = input_data.getShape();
        std::vector<int> target_shape = target_data.getShape();

        // Assuming num_samples is the first dimension of the input_data and target_data tensors
        int num_samples = input_shape[0];
        std::unique_ptr<Tensor<T>> sliced_tensor2; 
        Tensor<T> inpurt_data;
        Tensor<T>* predicted_output;
        for (int sample_index = 0; sample_index < num_samples; ++sample_index) {

            sliced_tensor2 = Hex::sliceFirstIndex(sample_index, input_data); 
            inpurt_data = *sliced_tensor2;
            predicted_output = &model.forward(inpurt_data, false);

            // Printing the predicted output
            std::cout << "Predicted output:" << std::endl;
            predicted_output->print();

            // Determine the actual output from the target_data
            int actual_output = (target_data.get({ sample_index, 0, 0 }) == 1) ? 0 : 1;

            // Print additional information
            if (predicted_output->get({ 0, 0 }) >= 0.5) {
                std::cout << "1st input for XOR: " << inpurt_data.get({ 0,0 }) << " 2nd input for XOR: " << inpurt_data.get({ 0,1 }) <<
                    " Neural Network output is 0, actual output is: " << actual_output << std::endl;
            }
            else {
                std::cout << "1st input for XOR: " << inpurt_data.get({ 0,0 }) << " 2nd input for XOR: " << inpurt_data.get({ 0,1 }) <<
                    " Neural Network output is 1, actual output is: " << actual_output << std::endl;
            }

            std::cout << "end of cycle" << std::endl;
            std::cout << std::endl;

            ////// Print the sliced tensor
            //std::cout << "Sliced tensor at index " << sample_index << ":" << std::endl;
            //sliced_tensor->print();

        }
    }

    template<typename T>
    void trainNeuralNetwork(MLP<T>& model, const Tensor<T>& input_data, const Tensor<T>& target_data, int num_epochs, T learning_rate) {
        // Get the number of samples
        std::vector<int> input_shape = input_data.getShape();

        int num_samples = input_shape[0];

        Tensor<T> sampled_input_data = input_data;
        Tensor<T> sampled_target_data = target_data;

      
 
        std::shared_ptr<Tensor<T>> predicted_output;

        std::shared_ptr<Tensor<T>> error;
        std::shared_ptr<Tensor<T>> output_error;

        sampled_input_data.reshape({ 4,2 });
        sampled_target_data.reshape({ 4,2 });

        // Training loop
        for (int epoch = 0; epoch < num_epochs; ++epoch) {

            T total_error = 0;
            for (int sample_index = 0; sample_index < num_samples; ++sample_index) {
             
                predicted_output = std::make_shared<Tensor<T>>(model.forward(sampled_input_data));
                
                error =  mse(sampled_target_data, *predicted_output);

                total_error += error->get({ 0 });
                //std::cout << total_error << std::endl;
                output_error =  mse_derivative(sampled_target_data, *predicted_output);
             
                model.backpropa(*output_error, learning_rate);
            }
            // Calculate the average error on all samples
            T average_error = (total_error / num_samples);
            std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs << "   Mean Squared Error: " << average_error << std::endl;
        }
        std::cout << std::endl;
    }


     void xor_example() {
        // Define the parameters for your MLP
        int input_size = 2;        // Size of input layer
        int output_size = 2;       // Size of output layer
        int batchsize = 4; 
        int h_l_dimension = 15;     // Dimension of each hidden layer

        // Create an instance of the MLP class
        std::unique_ptr<Hex::MLP<float>>  mlp(new  Hex::MLP<float>(input_size, output_size, batchsize,  h_l_dimension));


        // Define your input data
        std::vector<std::vector<std::vector<float>>> x_train = {
            {{0, 0}},
            {{0, 1}},
            {{1, 0}},
            {{1, 1}}
        };

        std::vector<std::vector<std::vector<float>>> y_train = {
            {{1, 0}},   // Class 0
            {{0, 1}},   // Class 1
            {{0, 1}},   // Class 1
             {{1, 0}}  // Class 0
        };

        // Create a Tensor for x_train
        std::vector<int> x_shape = { 4, 1, 2 }; // Shape: (4, 1, 2)
        std::unique_ptr<Tensor<float>> x_tensor(new Tensor<float>(x_shape));

        // Create a Tensor for y_train
        std::vector<int> y_shape = { 4, 1, 2 }; // Shape: (4, 1, 2)
        std::unique_ptr<Tensor<float>> y_tensor(new Tensor<float>(y_shape));
        // Set data for x_tensor
        for (int i = 0; i < 4; ++i) {
            x_tensor->set({ i, 0, 0 }, x_train[i][0][0]);
            x_tensor->set({ i, 0, 1 }, x_train[i][0][1]);

            y_tensor->set({ i, 0, 0 }, y_train[i][0][0]);
            y_tensor->set({ i, 0, 1 }, y_train[i][0][1]);
        }


        trainNeuralNetwork(*mlp, *x_tensor, *y_tensor, 100, 0.001f);
        predictAndPrintResults(*mlp, *x_tensor, *y_tensor);
    }


 

    template void predictAndPrintResults(MLP<float>& model, const Tensor<float>& input_data, const Tensor<float>& target_data);
    template void predictAndPrintResults(MLP<int>& model, const Tensor<int>& input_data, const Tensor<int>& target_data);
    template void predictAndPrintResults(MLP<double>& model, const Tensor<double>& input_data, const Tensor<double>& target_data);

    template void trainNeuralNetwork(MLP<float>& model, const Tensor<float>& input_data, const Tensor<float>& target_data, int num_epochs, float learning_rate) ;
    template void trainNeuralNetwork(MLP<int>& model, const Tensor<int>& input_data, const Tensor<int>& target_data, int num_epochs, int learning_rate) ;
    template void trainNeuralNetwork(MLP<double>& model, const Tensor<double>& input_data, const Tensor<double>& target_data, int num_epochs, double learning_rate) ;

}