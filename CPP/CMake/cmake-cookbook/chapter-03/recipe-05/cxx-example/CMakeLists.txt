# set minimum cmake version
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

# project name and language
project(recipe-05 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find openmp
find_package(OpenMP REQUIRED)

add_executable(example example.cpp)

# link openmp libraries to project,
# OpenMP::OpenMP_CXX contains all the infomation needed for opemmp integaration
target_link_libraries(example
  PUBLIC
    OpenMP::OpenMP_CXX
  )
