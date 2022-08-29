# so we don't need to include header files in this directory
set(CMAKE_INCLUDE_CURRENT_DIR_IN_INTERFACE ON)

# static library created named sum_integers
add_library(sum_integers sum_integers.cpp)

add_executable(sum_up main.cpp)

target_link_libraries(sum_up sum_integers)
