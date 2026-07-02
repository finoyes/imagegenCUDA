cmake_minimum_required(VERSION 3.20)
project(transformer_imggen LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set your GPU architecture: sm_86 = RTX 3000 series, sm_89 = RTX 4000 series
# Check yours with: nvidia-smi --query-gpu=compute_cap --format=csv
set(CMAKE_CUDA_ARCHITECTURES 86)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -lineinfo")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -O3 -Wall")

include_directories(vendor)


add_library(cuda_kernels STATIC
    cuda/kernels/matmul.cu
    cuda/kernels/softmax.cu
    cuda/kernels/layernorm.cu
    cuda/kernels/activations.cu
    cuda/kernels/attention.cu
)
set_target_properties(cuda_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(cuda_model STATIC
    cuda/model/encoder.cu
    cuda/model/decoder.cu
    cuda/model/transformer.cu
)
target_link_libraries(cuda_model cuda_kernels)
set_target_properties(cuda_model PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_library(cuda_training STATIC
    cuda/training/optimizer.cu
    cuda/training/loss.cu
    cuda/training/backward.cu
    cuda/training/scheduler.cpp
    cuda/utils/memory.cu
    cuda/utils/checkpoint.cpp
)
target_link_libraries(cuda_training cuda_model)


add_executable(train
    cpp/train.cpp
    cpp/dataloader.cpp
    cpp/logger.cpp
)
target_link_libraries(train cuda_training cuda_model cuda_kernels)


add_executable(preprocess
    cpp/preprocess.cpp
)


add_executable(generate
    cpp/generate.cpp
)
target_link_libraries(generate cuda_training cuda_model cuda_kernels)

cmake_minimum_required(VERSION 3.20)
project(transformer_imggen LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set your GPU architecture: sm_86 = RTX 3000 series, sm_89 = RTX 4000 series
# Check yours with: nvidia-smi --query-gpu=compute_cap --format=csv
set(CMAKE_CUDA_ARCHITECTURES 86)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -lineinfo")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -O3 -Wall")

# Vendor include path
include_directories(vendor)

# ---- CUDA kernel library ----
add_library(cuda_kernels STATIC
    cuda/kernels/matmul.cu
    cuda/kernels/softmax.cu
    cuda/kernels/layernorm.cu
    cuda/kernels/activations.cu
    cuda/kernels/attention.cu
)
set_target_properties(cuda_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ---- CUDA model library ----
add_library(cuda_model STATIC
    cuda/model/encoder.cu
    cuda/model/decoder.cu
    cuda/model/transformer.cu
)
target_link_libraries(cuda_model cuda_kernels)
set_target_properties(cuda_model PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ---- CUDA training library ----
add_library(cuda_training STATIC
    cuda/training/optimizer.cu
    cuda/training/loss.cu
    cuda/training/backward.cu
    cuda/training/scheduler.cpp
    cuda/utils/memory.cu
    cuda/utils/checkpoint.cpp
)
target_link_libraries(cuda_training cuda_model)

# ---- Trainer executable ----
add_executable(train
    cpp/train.cpp
    cpp/dataloader.cpp
    cpp/logger.cpp
)
target_link_libraries(train cuda_training cuda_model cuda_kernels)

# ---- Preprocessor executable ----
add_executable(preprocess
    cpp/preprocess.cpp
)

# ---- Generator executable ----
add_executable(generate
    cpp/generate.cpp
)
target_link_libraries(generate cuda_training cuda_model cuda_kernels)
