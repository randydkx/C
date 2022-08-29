# set minimum cmake version
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# project name and language
project(recipe-01 LANGUAGES NONE)

# detect python
# required 表示如果没找到对应的package就强制表示已经satisfied
find_package(PythonInterp REQUIRED)
# cmake -D PYTHON_EXECUTABLE=/custom/location/python .. 可以直接指定python解释器的位置

# Execute a tiny Python script
execute_process(
  COMMAND
    ${PYTHON_EXECUTABLE} "-c" "print('Hello, world!')"
  RESULT_VARIABLE _status
  OUTPUT_VARIABLE _hello_world
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )

message(STATUS "RESULT_VARIABLE is: ${_status}")
message(STATUS "OUTPUT_VARIABLE is: ${_hello_world}")

# compare the "manual" messages with the following handy helper
include(CMakePrintHelpers)
cmake_print_variables(_status _hello_world)
