cmake_minimum_required(VERSION 2.8.12)
project(example)

set (CMAKE_CXX_STANDARD 14)

find_package( EIGEN3 REQUIRED)
find_package(pybind11 REQUIRED)
include_directories( ${EIGEN3_INCLUDE_DIRS} )

pybind11_add_module(example example.cpp)
