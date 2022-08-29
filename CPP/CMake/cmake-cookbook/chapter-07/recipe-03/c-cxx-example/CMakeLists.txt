cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(recipe-03 LANGUAGES C CXX)
# 包含一个指定的cmake文件
include(set_compiler_flag.cmake)

# working_compile_flag是在set_compiler_flag.cmake文件夹中定义的
set_compiler_flag(
  working_compile_flag C REQUIRED
  "-foo"             # this should fail
  "-wrong"           # this should fail
  "-wrong"           # this should fail
  "-Wall"            # this should work with GNU
  "-warn all"        # this should work with Intel
  "-Minform=inform"  # this should work with PGI
  "-nope"            # this should fail
  )

message(STATUS "working C compile flag: ${working_compile_flag}")

set_compiler_flag(
  working_compile_flag CXX REQUIRED
  "-foo"    # this should fail
  "-g"      # this should work with GNU, Intel, PGI
  "/RTCcsu" # this should work with MSVC
  )

message(STATUS "working CXX compile flag: ${working_compile_flag}")
