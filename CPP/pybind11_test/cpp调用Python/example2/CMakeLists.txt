# 
# 用于在C++中执行Python相关的操作，需要包含Python头文件的路径以及lib路径还有pybind11头文件的路径
# 
cmake_minimum_required(VERSION 3.0.4)
project(example)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(PROJECT_NAME example)


# 使用指定版本的Python解释器，3.8.5
find_package(PythonInterp REQUIRED HINT /usr/local/bin)

# 查找pybind11相关的库
include_directories(/usr/local/include)

add_executable(${PROJECT_NAME} example.cpp)

# 指定Python头文件的路径 + Python库的搜索路径
set(Python3_INCLUDE_DIRS /Users/wenshuiluo/opt/anaconda3/include/python3.8)
set(Python3_LIBRARIES /Users/wenshuiluo/opt/anaconda3/lib/libpython3.8.dylib)

# 查找Python.h相关的库
target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS})
# 查找Python依赖的库文件
target_link_libraries(${PROJECT_NAME} PRIVATE ${Python3_LIBRARIES})

message(STATUS "python library directory:${Python3_LIBRARIES}")
message(STATUS "python include directory:${Python3_INCLUDE_DIRS}")