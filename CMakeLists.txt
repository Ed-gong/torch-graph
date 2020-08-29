cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(custom_class)
set(CMAKE_CXX_STANDARD 14)
find_package(Torch REQUIRED)
#message(STATUS "${TORCH_LIBRARIES}")

set(GONE_DIR "/home/datalab/src/dist-graph")

include_directories("${GONE_DIR}/src" 
                    "${GONE_DIR}/onedata" 
                    "${GONE_DIR}/dist"
                    "${GONE_DIR}/gview"
                    "${GONE_DIR}/analytics"
	)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DB32 -DOVER_COOMIT -DPLAIN_GRAPH -DTBB -DDEL -DSDEL -DCOMPACTION -fopenmp"
)



# Define our library target
add_library(dcgan SHARED  main.cpp GCN.h GCN.cpp ManagerWrap.h ManagerWrap.cpp SnapWrap.h SnapWrap.cpp)

# Link against LibTorch
target_link_libraries(dcgan "${TORCH_LIBRARIES}")
target_link_libraries(dcgan -Wl,--no-as-needed ${GONE_DIR}/libgraphone32.a -ltbb) 
#target_link_libraries(dcgan "${SOURCES}")

