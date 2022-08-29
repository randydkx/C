# CMake 最低版本号要求
cmake_minimum_required (VERSION 3.10)
# 项目信息
project (Demo1)
# 设置静态链接库、动态链接库和可执行文件的执行路径
# archive表示静态链接库.a文件 library表示动态链接库.dylib文件 runtime表示可执行文件
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 指定生成目标
add_executable(Demo test.cpp)
