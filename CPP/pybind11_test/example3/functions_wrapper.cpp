#include <pybind11/pybind11.h> 
#include "functions.h" 
 
namespace py = pybind11; 
 
PYBIND11_MODULE(functions, m){ 
 m.doc() = "Simple Class"; 
 m.def("add", &add); 
 m.def("sub", &sub); 
 m.def("mul", &mul); 
 m.def("div", &div1); 
}