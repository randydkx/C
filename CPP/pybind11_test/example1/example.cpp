#include <pybind11/pybind11.h>
#include<iostream>

namespace py = pybind11;
using namespace pybind11::literals;

// 功能函数
int add(int i = 1, int j = 2) {
    // print();
    return i + j;
}

// int main(){
//     std::cout<<"hello world"<<std::endl;
// }

// python绑定代码
PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.attr("the_answer") = 42;
    py::object world = py::cast("World");
    m.attr("World") = world;
    m.def("add", &add, "A function which adds two numbers", "i"_a = 1, "j"_a = 2);
}
