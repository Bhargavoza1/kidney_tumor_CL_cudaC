cmake_minimum_required(VERSION 3.13)
project(cuda_nn)

# Find CUDA package
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
list(APPEND CUDA_NVCC_FLAGS -arch=sm_86)

# Find OpenCV package


# Include directories for CUDA, OpenCV, and other headers
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/costs
    ${CMAKE_CURRENT_SOURCE_DIR}/layers
    ${CMAKE_CURRENT_SOURCE_DIR}/models
)

# Add executable target
set(SOURCES
    app.cpp
    costs/MSE.cu
    example/XOR.cu
    example/kidney_cf.cpp
    layers/linear.cu
    layers/ReLU.cu
    layers/CNN2D.cu
    layers/Sigmoid.cu
    layers/MaxPool2d.cu
    layers/flatten_layer.cu
    layers/BatchNorm.cu
    models/MLP.cpp
    models/Image_CF.cpp
    utils/Errorhelper.cpp
    utils/Tensor.cu
    utils/tensor_oprations.cu
)

# Set CUDA language for CUDA sources
#cuda_add_executable(cuda_nn ${SOURCES})
cuda_add_library(app  SHARED ${SOURCES})
# Link CUDA and OpenCV libraries
target_link_libraries(app stdc++  ${OpenCV_LIBS} ${CUDA_LIBRARIES} )

# Expose include directory
target_include_directories(app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Install the library
install(TARGETS app DESTINATION lib)
install(FILES app.h DESTINATION include)

# Command to compile the Go program
set(GO_SOURCE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/app.go)
# Print the value of GO_SOURCE_FILE
message("Go source file: ${GO_SOURCE_FILE}")

# Set the output path for the executable
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

# Invoke the Go compiler directly using a custom command
add_custom_command(
    OUTPUT backend
    COMMAND go build -o backend ${GO_SOURCE_FILE}
    DEPENDS ${GO_SOURCE_FILE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Compiling Go program"
)

# Add a custom target to encapsulate the Go compilation
add_custom_target(BuildGoProgram ALL DEPENDS backend)

# Add the compiled executable as a target
#add_executable(GoExample IMPORTED GLOBAL)
#set_target_properties(GoExample PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/backend)
#add_dependencies(GoExample BuildGoProgram)

# have to export build dir or dir of .so to run programme
#export LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH