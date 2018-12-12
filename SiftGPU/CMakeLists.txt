
cmake_minimum_required(VERSION 2.8.3)

project(test_siftgpu)

set(CMAKE_VERBOSE_MAKEFILE on)

set(OpenCV_DIR "/usr/local/opencv3.4.4/share/OpenCV")
find_package(OpenCV REQUIRED)

find_package(OpenGL REQUIRED)

find_package(GLUT REQUIRED)

#find_package(Glew REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# set siftgpu
include_directories("/home/liqiang/Downloads/SiftGPU/src/SiftGPU")
include_directories(${OpenGL_INCLUDE_DIR})

link_directories(/usr/lib64) # GLEW

set(SIFTGPU_LIBS "/home/liqiang/Downloads/SiftGPU/bin/libsiftgpu.so")

add_executable(testSiftGPU main.cc)

target_link_libraries(testSiftGPU ${OpenCV_LIBS} ${SIFTGPU_LIBS} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES})