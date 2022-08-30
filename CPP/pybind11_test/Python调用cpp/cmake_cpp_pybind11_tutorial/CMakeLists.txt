cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
# project name
project(automobile)

# include source directories
include_directories("${CMAKE_SOURCE_DIR}/cpp/include/automobile_bits")
include_directories("${CMAKE_SOURCE_DIR}/python")

# 配置python库的搜索路径以及python解释器的路径
set(DPYTHON_LIBRARY_DIR /Users/wenshuiluo/opt/anaconda3/lib/python3.8/site-packages)
set(DPYTHON_EXECUTABLE /Users/wenshuiluo/opt/anaconda3/bin/python)

file (GLOB SOURCE_FILES "cpp/src/*.cpp")
file (GLOB HEADER_FILES "cpp/include/automobile_bits/*.hpp")
file (GLOB PYTHON_FILES "python/*.cpp" "python/*.hpp")

# 已经install过pybind11，通过指令查找即可
find_package(pybind11 REQUIRED)
pybind11_add_module(automobile ${SOURCE_FILES} ${HEADER_FILES} ${PYTHON_FILES})

# 链接相关的库
target_link_libraries(automobile PUBLIC)

# Library destination为.so库的安装路径，安装完成之后可以在任何位置使用import得到编译的so库
install(TARGETS automobile LIBRARY DESTINATION ${DPYTHON_LIBRARY_DIR})
