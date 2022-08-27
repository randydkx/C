#include <pybind11/pybind11.h>

namespace py=pybind11;
using namespace std;
class Pet {
public:
    Pet(const string &name) : name(name) { }
    void setName(const string &name_) { name = name_; }
    const string &getName() const { return name; }
    string name;
};

PYBIND11_MODULE(example3, m) {
    py::class_<Pet>(m, "Pet")
        .def(py::init<const std::string &>())
        .def("setName", &Pet::setName)
        .def("getName", &Pet::getName);
}
