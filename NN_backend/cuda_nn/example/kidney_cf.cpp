#include "kidney_cf.h"
#include <iostream> 
 
 
using namespace std;

namespace Hex {

    std::string normalPath = "../kidney-ct-scan-image/Normal/";
    std::string tumorPath = "../kidney-ct-scan-image/Tumor/";

    
    std::vector<cv::String> trainFilePaths;
    std::vector<cv::String> testFilePaths;

    std::vector<cv::Mat> train_Images;
    std::vector<std::vector<int>>  train_LabelsOneHot;

    std::vector<cv::Mat> test_Images;
    std::vector<std::vector<int>>  test_LabelsOneHot;

    int resizeWidth = 225;
    int resizeHeight = 225;
    int channels = 3;
    int batchSize = 8;
    int epoch = 20;


    void trainTestSplit(const std::vector<cv::String>& allFilePaths, float trainRatio,
        std::vector<cv::String>& trainFilePaths, std::vector<cv::String>& testFilePaths) {
        // Shuffle the file paths
        std::vector<cv::String> shuffledFilePaths = allFilePaths;
        std::shuffle(shuffledFilePaths.begin(), shuffledFilePaths.end(), std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count()));

        // Calculate split indices
        size_t splitIndex = static_cast<size_t>(trainRatio * shuffledFilePaths.size());

        // Assign training and testing file paths
        trainFilePaths.assign(shuffledFilePaths.begin(), shuffledFilePaths.begin() + splitIndex);
        testFilePaths.assign(shuffledFilePaths.begin() + splitIndex, shuffledFilePaths.end());
    }

    void imagepreprocess(int width, int height, std::string normalPath, std::vector<cv::String> filepaths, std::vector<std::vector<int>>& lable, std::vector<cv::Mat>& images) {

        for (const auto& filePath : filepaths) {
            cv::Mat image = cv::imread(filePath, cv::IMREAD_GRAYSCALE);
            if (image.empty()) {
                std::cerr << "Failed to load image: " << filePath << std::endl;
                return;
            }

            // Resize the image to the specified width and height
            cv::resize(image, image, cv::Size(width, height));

            // Convert single channel grayscale to 3-channel grayscale
            cv::Mat colorImage;
            cv::cvtColor(image, colorImage, cv::COLOR_GRAY2BGR);

            // Normalize pixel values to the range [0, 1]
            cv::Mat normalizedImage;
            colorImage.convertTo(normalizedImage, CV_32FC3, 1.0 / 255.0);

            // Determine label based on folder
            std::string formattedPath = filePath;
            std::replace(formattedPath.begin(), formattedPath.end(), '\\', '/');
            int label = (formattedPath.find(normalPath) != std::string::npos) ? 0 : 1;
            // One-hot encode the label
            std::vector<int> labelOneHot(2, 0); // Initialize one-hot encoded label with zeros
            labelOneHot[label] = 1; // Set the appropriate index to 1 based on the label
            lable.push_back(labelOneHot); // Push the one-hot encoded label into the vector

            images.push_back(normalizedImage);

        }

    }

    void trainNeuralNetwork2(Image_CF<float>& model, std::vector<cv::Mat>& input, std::vector<std::vector<int>>& target,
        int batch, int channels, int num_epochs, float learning_rate)
    {
        // Create a Tensor object to store the images 
        int numImages = input.size();
        int height = input[0].rows;
        int width = input[0].cols;

        int numBatches = (numImages + batch - 1) / batch;

        std::vector<int> shape = { batch , channels , height, width };
        Hex::Tensor<float> imageTensor(shape);
        Hex::Tensor<float> labelTensor({ batch,2 });

        // Copy image data from CPU to GPU
        size_t size = batch * height * width * channels * sizeof(float);
        float* gpuData;
        cudaError_t cudaStatus = cudaMalloc((void**)&gpuData, size);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
            return;
        }

        size_t labelSize = batch * 2 * sizeof(float);
        float* gpuLabelData;
        cudaError_t cudaStatusLabel = cudaMalloc((void**)&gpuLabelData, labelSize);
        if (cudaStatusLabel != cudaSuccess) {
            std::cerr << "cudaMalloc for label tensor failed: " << cudaGetErrorString(cudaStatusLabel) << std::endl;
            cudaFree(gpuLabelData); // Free the previously allocated memory
            return;
        }
        std::shared_ptr<Tensor<float>> error;
        std::shared_ptr<Tensor<float>> output_error;
        Tensor<float> a;

        for (int e = 0; e < num_epochs; e++) {
            float total_error = 0;
            for (int i = 0; i < numBatches; ++i) {
                for (int j = 0; j < batch; ++j) {
                    int index = i * batch + j;
                    if (index < numImages) {
                        cudaStatus = cudaMemcpy(gpuData + j * height * width * channels, input[index].data, size / batch, cudaMemcpyHostToDevice);
                        if (cudaStatus != cudaSuccess) {
                            std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(cudaStatus) << std::endl;
                            cudaFree(gpuData);
                            return;
                        }

                        for (int k = 0; k < target[index].size(); ++k) {
                            float labelValue = static_cast<float>(target[index][k]);
                            cudaStatus = cudaMemcpy(gpuLabelData + j * 2 + k, &labelValue, sizeof(float), cudaMemcpyHostToDevice);
                            if (cudaStatus != cudaSuccess) {
                                std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(cudaStatus) << std::endl;
                                cudaFree(gpuData);
                                cudaFree(gpuLabelData);
                                return;
                            }
                        }
                    }
                }


                imageTensor.setData(gpuData);

                labelTensor.setData(gpuLabelData);



                a = model.forward(imageTensor);
                error = Hex::mse(labelTensor, a);

                total_error += error->get({ 0 });

                output_error = Hex::mse_derivative(labelTensor, a);
                model.backpropa(*output_error, learning_rate);

            }
            float average_error = (total_error / numBatches);
            std::cout << "Epoch " << (e + 1) << "/" << num_epochs << "   Mean Squared Error: " << average_error << std::endl;
            // a.print();
             //labelTensor.print();

            // std::cout << std::endl;
        }

        cudaFree(gpuData);
        cudaFree(gpuLabelData);

    }

    void testNeuralNetwork2(Image_CF<float>& model, std::vector<cv::Mat>& input, std::vector<std::vector<int>>& target, std::vector<cv::String> filepaths)
    {

        // Create a Tensor object to store the images 
        int numImages = input.size();
        int height = input[0].rows;
        int width = input[0].cols;
        int channels = 3;
        int batch = 1;

        std::vector<int> shape = { batch , channels , height, width };
        Hex::Tensor<float> imageTensor(shape);
        Hex::Tensor<float> labelTensor({ batch,2 });

        // Copy image data from CPU to GPU
        size_t size = height * width * channels * sizeof(float);
        float* gpuData;
        cudaError_t cudaStatus = cudaMalloc((void**)&gpuData, size);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
            return;
        }

        size_t labelSize = batch * 2 * sizeof(float);
        float* gpuLabelData;
        cudaError_t cudaStatusLabel = cudaMalloc((void**)&gpuLabelData, labelSize);
        if (cudaStatusLabel != cudaSuccess) {
            std::cerr << "cudaMalloc for label tensor failed: " << cudaGetErrorString(cudaStatusLabel) << std::endl;
            cudaFree(gpuLabelData); // Free the previously allocated memory
            return;
        }

        Tensor<float> a;

        for (int image_x = 0; image_x < numImages; ++image_x) {
            cudaStatus = cudaMemcpy(gpuData, input[image_x].data, size, cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(cudaStatus) << std::endl;
                cudaFree(gpuData);
                return;
            }

            for (int k = 0; k < target[image_x].size(); ++k) {
                float labelValue = static_cast<float>(target[image_x][k]);
                cudaStatus = cudaMemcpy(gpuLabelData + k, &labelValue, sizeof(float), cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(cudaStatus) << std::endl;
                    cudaFree(gpuData);
                    cudaFree(gpuLabelData);
                    return;
                }
            }

            imageTensor.setData(gpuData);

            labelTensor.setData(gpuLabelData);

            //imageTensor.printshape();
             //labelTensor.print();


            std::cout << "file path of image" << endl;
            std::cout << filepaths[image_x] << endl;
            std::cout << "actual value : " << endl;
            labelTensor.print();
            a = model.forward(imageTensor, false);
            std::cout << "predicted value : " << endl;
            a.print();


            std::cout << std::endl;
        }



    }

    void processImagesAndRunNeuralNetwork()
    {
 

        if(!Imagecf){

             // Load file paths from the Normal directory
        std::vector<cv::String> normalFilePaths;
        cv::glob(normalPath + "*.jpg", normalFilePaths);

        // Load file paths from the Tumor directory
        std::vector<cv::String> tumorFilePaths;
        cv::glob(tumorPath + "*.jpg", tumorFilePaths);

        // Combine all file paths into one vector
        std::vector<cv::String> allFilePaths;
        allFilePaths.insert(allFilePaths.end(), normalFilePaths.begin(), normalFilePaths.end());
        allFilePaths.insert(allFilePaths.end(), tumorFilePaths.begin(), tumorFilePaths.end());



        float trainRatio = 0.981;
   

        // Perform train-test split
        trainTestSplit(allFilePaths, trainRatio, trainFilePaths, testFilePaths);

 


        /////imagepreprocess for train data
        imagepreprocess(resizeWidth, resizeHeight, normalPath, trainFilePaths, train_LabelsOneHot, train_Images);

        /////imagepreprocess for test data
        imagepreprocess(resizeWidth, resizeHeight, normalPath, testFilePaths, test_LabelsOneHot, test_Images);

 

            Imagecf =  new Image_CF<float>(  batchSize, channels, 2 );
        }

        trainNeuralNetwork2(*Imagecf, train_Images, train_LabelsOneHot, batchSize, channels, epoch, 0.00001f);
        std::cout << std::endl;
        std::cout << std::endl;

       
    }

    void test(){
        testNeuralNetwork2(*Imagecf, test_Images, test_LabelsOneHot, testFilePaths);
    }

    int predict() {
        string filePath = "./kidny.jpg";
            cv::Mat image = cv::imread(filePath, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << filePath << std::endl;
          
        }
        int width = 225;
        int height = 225;
        int channels = 3;
        // Resize the image to the specified width and height
        cv::resize(image, image, cv::Size(width, height));

        // Convert single channel grayscale to 3-channel grayscale
        cv::Mat colorImage;
        cv::cvtColor(image, colorImage, cv::COLOR_GRAY2BGR);

        // Normalize pixel values to the range [0, 1]
        cv::Mat normalizedImage;
        colorImage.convertTo(normalizedImage, CV_32FC3, 1.0 / 255.0);

        // Determine label based on folder
        //int height = normalizedImage.rows;
        //int width = normalizedImage.cols;
 
        int batch = 1;

        std::vector<int> shape = { batch , channels , height, width };
        Hex::Tensor<float> imageTensor(shape);
        //normalizedImage

        // Copy image data from CPU to GPU
        size_t size = height * width * channels * sizeof(float);
        float* gpuData;
        cudaError_t cudaStatus = cudaMalloc((void**)&gpuData, size);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
            
        }

        size_t labelSize = batch * 2 * sizeof(float);
        float* gpuLabelData;
        cudaError_t cudaStatusLabel = cudaMalloc((void**)&gpuLabelData, labelSize);
        if (cudaStatusLabel != cudaSuccess) {
            std::cerr << "cudaMalloc for label tensor failed: " << cudaGetErrorString(cudaStatusLabel) << std::endl;
            cudaFree(gpuLabelData); // Free the previously allocated memory
            
        }

        Tensor<float> a;

     
            cudaStatus = cudaMemcpy(gpuData, normalizedImage.data, size, cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(cudaStatus) << std::endl;
                cudaFree(gpuData);
                 
            }

            

            imageTensor.setData(gpuData);

         
            //imageTensor.printshape();
             //labelTensor.print();


 
            a = Imagecf->forward(imageTensor, false);
            std::cout << "predicted value : " << endl;
            a.print();


            std::cout << std::endl;
      
        return a.get({ 0, 0 }) > a.get({ 0, 1 }) ? 0 : 1 ;
        
    }
}