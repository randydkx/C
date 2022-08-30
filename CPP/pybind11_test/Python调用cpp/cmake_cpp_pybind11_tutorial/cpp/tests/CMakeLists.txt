cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(test_cpp)
# c++依赖库文件的搜索路径
include_directories(/usr/local/include)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/../bin)

# 检索dylib（named automobile) 并标记为AUTOMOBILE_LIB
# hint为推荐的搜索路径
find_library(AUTOMOBILE_LIB automobile HINTS /Users/wenshuiluo/coding/CPP/pybind11_test/cmake_cpp_pybind11_tutorial/cpp/build)

add_executable(test_cpp src/test_cpp.cpp)

target_link_libraries(test_cpp PUBLIC ${AUTOMOBILE_LIB})