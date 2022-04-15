#include <pybind11/pybind11.h>

int add( int i, int j ){
    return i+j;
}

PYBIND11_MODULE( py2cpp, m ){
    m.doc() = "pybind11 example";
    m.def("add", &add, "add two number" );
}