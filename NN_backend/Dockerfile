
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

 
  
WORKDIR /app


# Copy the contents of the project directory into the container
COPY . /app
# Use the official Ubuntu 22.04 image as the base image

 
RUN apt-get update

# Install necessary dependencies
RUN apt-get install -y wget \
                   build-essential \
                   cmake \
                   git \
                   pkg-config \
                   libjpeg-dev \
                   libpng-dev \
                   libopenjp2-7-dev \
                   golang

 
#RUN git config --global http.postBuffer 524288000
# Clone OpenCV repository
RUN git clone -b 4.9.0 https://github.com/opencv/opencv.git


 
# Create build directory and move into it
RUN cd /app/opencv && \
    mkdir build && \
    cd build && \
    cmake -D BUILD_TIFF=OFF \
          -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D BUILD_opencv_world=ON \
          -D BUILD_opencv_python=OFF \
          -D BUILD_opencv_java=OFF \
          -D BUILD_opencv_python2=OFF \
          -D BUILD_opencv_python3=OFF \
          -D BUILD_opencv_apps=OFF \
          -D BUILD_opencv_calib3d=OFF \
          -D BUILD_opencv_features2d=OFF \
          -D BUILD_opencv_flann=OFF \
          -D BUILD_opencv_highgui=OFF \
          -D BUILD_opencv_imgcodecs=ON \
          -D BUILD_opencv_imgproc=ON \
          -D BUILD_opencv_ml=OFF \
          -D WITH_IPP=OFF \
          -D BUILD_opencv_objdetect=OFF \
          -D BUILD_opencv_photo=OFF \
          -D BUILD_opencv_stitching=OFF \
          -D BUILD_opencv_video=OFF \
          -D BUILD_opencv_videoio=OFF \
          -D WITH_GTK=OFF \
          -D BUILD_opencv_dnn=OFF \
          -D BUILD_opencv_gapi=OFF \
          -D WITH_EIGEN=OFF ..



# Compile OpenCV with 8 parallel jobs
RUN cd opencv/build && make -j$(nproc) && make install && ldconfig

 
ENV CPLUS_INCLUDE_PATH=/usr/local/include/opencv4:$CPLUS_INCLUDE_PATH

RUN cd /app/cuda_nn && mkdir build && \
    cd build && \
    cmake .. && \
    make

    
# Set the working directory inside the container
WORKDIR /app/cuda_nn/build

# Expose port 8080
EXPOSE 8080

ENV LD_LIBRARY_PATH=/app/cuda_nn/build:$LD_LIBRARY_PATH
# Set the entry point to the backend executable
CMD ["./backend"]
#CMD ["bash", "-c", "while true; do sleep 3600; done"]

#docker run --gpus all -it -p 8080:8080  your_image_name