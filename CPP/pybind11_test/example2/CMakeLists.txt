cmake_minimum_required(VERSION 3.0.4)
project(example)

# 添加pybind11依赖
# set(pybind11_DIR /usr/local/bin/pybind11)
find_package(pybind11 REQUIRED)
message(STATUS "pybind11 version: ${pybind11_VERSION}")

# 添加额外的源文件
aux_source_directory(./src SRC)

# 添加python包
pybind11_add_module(example ${SRC}) 
