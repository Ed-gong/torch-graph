cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(custom_class)
set(CMAKE_CXX_STANDARD 14)
find_package(Torch REQUIRED)

include_directories()

set(GONE_DIR "/home/datalab/src/dist-graph")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DB32 -DOVER_COOMIT -DTBB -DDEL -DSDEL")

# Define our library target
add_library(custom_class SHARED dcgan.cpp)

# Link against LibTorch
target_link_libraries(dcgan ${GONE_DIR}/libgraphone32.a "${TORCH_LIBRARIES}" -ltbb)
