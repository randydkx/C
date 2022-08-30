#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <fstream>
#include <iostream>
// #include<iostream>
// #include<cstdio>
namespace py = pybind11;
using namespace pybind11::literals;

int main()
{
    Py_Initialize();
    py::module_ os = py::module_::import("os");
    py::module_ path = py::module_::import("os.path"); // like 'import os.path as path'
    py::module_ np = py::module_::import("numpy");     // like 'import numpy as np'

    py::str curdir_abs = path.attr("abspath")(path.attr("curdir"));
    py::print(py::str("Current directory: ") + curdir_abs);
    py::dict environ = os.attr("environ");
    py::print(environ["HOME"]);
    py::array_t<float> arr = np.attr("ones")(3, "dtype"_a = "float32");
    py::print(py::repr(arr + py::int_(1)));

    // 使用python命令行运行C++程序
    // py::str result = os.attr("system")(py::str("cd '/Users/wenshuiluo/coding/CPP/testFile/' && g++ STLtest.cpp -o STLtest.out -std=c++11 && '/Users/wenshuiluo/coding/CPP/testFile/'STLtest.out"));
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\".\")");
    // 使用python命令行执行python程序，然后可以通过文件进行交互
    os.attr("system")(py::str("cd '/Users/wenshuiluo/coding/CPP/pybind11_test/example3' && python -u run.py"));

    std::string line;
    std::ifstream is;
    // 推荐使用绝对路径/Users/wenshuiluo/coding/CPP/pybind11_test/example3/input.txt
    // 当前执行的目录如果是build目录则使用../获得input所在目录，如果不是build目录则直接使用input.txt获取
    is.open("../input.txt");
    std::cout<<is.is_open()<<std::endl;
    if (is.is_open())
    {
        getline(is, line);
        std::cout<<"reading..."<<std::endl;
        std::cout << line << std::endl;
    }

    // py::print(result);
}