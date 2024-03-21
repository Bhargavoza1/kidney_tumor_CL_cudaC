#pragma once
 
#include <iostream>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <algorithm> // for std::shuffle
#include <random>    // for std::default_random_engine
#include <chrono>    // for std::chrono::system_clock

#include "../models/Image_CF.h"
#include "../utils/Tensor.h"
#include "../utils/tensor_oprations.h"
#include "../layers/BatchNorm.h"
#include "../layers/linear.h"
#include "../models/MLP.h"
#include "../costs/MSE.h"
#include "../layers/CNN2D.h"


namespace Hex {
	static Image_CF<float>*   Imagecf;
	
	void trainTestSplit(const std::vector<cv::String>& allFilePaths, float trainRatio,
		std::vector<cv::String>& trainFilePaths, std::vector<cv::String>& testFilePaths);

	void imagepreprocess(int width, int height, std::string normalPath, std::vector<cv::String> filepaths, std::vector<std::vector<int>>& lable, std::vector<cv::Mat>& images);

	void trainNeuralNetwork2(Image_CF<float>& model, std::vector<cv::Mat>& input, std::vector<std::vector<int>>& target,
		int batch, int channels, int num_epochs, float learning_rate);
	void testNeuralNetwork2(Image_CF<float>& model, std::vector<cv::Mat>& input, std::vector<std::vector<int>>& target, std::vector<cv::String> filepaths);

	void processImagesAndRunNeuralNetwork();
	void test();
	int predict();
}