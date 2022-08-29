cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# language None  表示不需要语言的支持
project(recipe-02 LANGUAGES NONE)

# CMKAE_MODULE_PATH用于对cmake 模块进行检索（for include or find_package）
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# 添加一个cmake文件
include(colors)
define_colors()

message(STATUS "This is a normal message")
message(STATUS "${Red}This is a red${ColourReset}")
message(STATUS "${BoldRed}This is a bold red${ColourReset}")
message(STATUS "${Green}This is a green${ColourReset}")
message(STATUS "${BoldMagenta}This is bold${ColourReset}")
