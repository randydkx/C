#include <iostream>
#include <pybind11/pybind11.h>

using namespace std;
namespace py = pybind11;

// 自定义结构体
struct Pet {
    Pet(const string &name) : name(name) { }
    void setName(const string &name_) { name = name_; }
    const string &getName() const { return name; }

    string name;
};


// 绑定代码
PYBIND11_MODULE(example2, m) {
    py::class_<Pet>(m, "Pet")
        .def(py::init<const std::string &>())
        .def("setName", &Pet::setName)
        .def("getName", &Pet::getName);
}
