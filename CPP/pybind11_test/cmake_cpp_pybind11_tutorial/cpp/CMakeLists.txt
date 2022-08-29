cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(automobile VERSION 0.1.0)

# 头文件的搜索路径
include_directories(/usr/local/include)

# Src
AUX_SOURCE_DIRECTORY(src SRC_FILES)

# Headers
set(PROJECT_SOURCE_DIR "src")
set(PROJECT_INCLUDE_DIR "include/automobile_bits")

# Source files
set(SOURCE_FILES
    ${PROJECT_INCLUDE_DIR}/motorcycle.hpp
    ${PROJECT_SOURCE_DIR}/motorcycle.cpp
)

# Set up such that XCode organizes the files correctly
# source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES})

# Add library
add_library(automobile SHARED ${SOURCE_FILES})

# Include directories
# these cpp files 
target_include_directories(automobile PRIVATE include/automobile_bits)

# Install the dylib automobile
install(TARGETS automobile DESTINATION lib)

# Install the header file 
install(FILES include/automobile DESTINATION include)

# Create base directory for header file above
install(DIRECTORY include/automobile_bits DESTINATION include)