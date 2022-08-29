cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-06 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置头文件的额外搜索路径
include_directories(/usr/local/include)
find_package(eigen3 3.4.0 REQUIRED CONFIG)

include(CheckCXXCompilerFlag)

# 检查是否支持某些compile flag
# 第一个参数是测试的compile flag，第二个参数是保存的结果，false or true
check_cxx_compiler_flag("-march=native" _march_native_works)
check_cxx_compiler_flag("-xHost" _xhost_works)

set(_CXX_FLAGS)
if(_march_native_works)
  message(STATUS "Using processor's vector instructions (-march=native compiler flag set)")
  set(_CXX_FLAGS "-march=native")
elseif(_xhost_works)
  message(STATUS "Using processor's vector instructions (-xHost compiler flag set)")
  set(_CXX_FLAGS "-xHost")
else()
  message(STATUS "No suitable compiler flag found for vectorization")
endif()

add_executable(linear-algebra-unoptimized linear-algebra.cpp)

target_link_libraries(linear-algebra-unoptimized
  PRIVATE
    Eigen3::Eigen
  )

add_executable(linear-algebra linear-algebra.cpp)

# 添加了向量化并行的指令
target_compile_options(linear-algebra
  PRIVATE
    ${_CXX_FLAGS}
  )

target_link_libraries(linear-algebra
  PRIVATE
    Eigen3::Eigen
  )
