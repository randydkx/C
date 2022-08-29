#include <iostream>
#include <pybind11/pybind11.h>

using namespace std;
namespace py = pybind11;

// 自定义结构体
struct Pet {
public:
    Pet(const string &name) : name(name) { }
    void setName(const string &name_) { name = name_; }
    const string &getName() const { return name; }
private:
    string name;
};


// 绑定代码
PYBIND11_MODULE(example2, m) {
    // dynamic_attr to ensure add new properties when an instance has been created
    py::class_<Pet>(m, "Pet", py::dynamic_attr())
        .def(py::init<const std::string &>())
        .def("setName", &Pet::setName)
        .def("getName", &Pet::getName)
        // expose name field to ensure read & write
        // .def_readwrite("name", &Pet::name)
        // for readable and writable property using setter and getter
        .def_property("name",&Pet::getName, &Pet::setName)
        .def("__repr__",
            [](const Pet &a) {
                return "<example.Pet named '" + a.getName() + "'>";
            }
        );
}
