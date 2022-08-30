cmake_minimum_required(VERSION 2.8.12)
project(example)

find_package(pybind11 REQUIRED)
find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
message("eigen3 path : ${EIGEN3_INCLUDE_DIRS}")
include_directories( ${EIGEN3_INCLUDE_DIRS} )

pybind11_add_module(example example.cpp)
