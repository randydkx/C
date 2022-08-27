#include <pybind11/pybind11.h>
#include<iostream>
// #include<test.h>
namespace py = pybind11;

// 功能函数
int add(int i, int j) {
    // print();
    return i + j;
}

// int main(){
//     std::cout<<"hello world"<<std::endl;
// }

// python绑定代码
PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
}
