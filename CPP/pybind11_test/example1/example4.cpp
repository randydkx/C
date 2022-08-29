#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <iostream>

namespace py = pybind11;
using namespace py::literals;

// 使用array_t操作numpy类型
py::array_t<double> add_c(py::array_t<double> arr1, py::array_t<double> arr2) {
    py::buffer_info buf1 = arr1.request(), buf2 = arr2.request();
    if (buf1.shape != buf2.shape)
        throw std::runtime_error("Input shapes must match");
    /* No pointer is passed, so NumPy will allocate the buffer */
    auto result = py::array_t<double>(buf1);
    py::buffer_info buf3 = result.request();
    double* ptr1 = (double*)buf1.ptr,
        * ptr2 = (double*)buf2.ptr,
        * ptr3 = (double*)buf3.ptr;
    #pragma omp parallel for
    for (ssize_t idx = 0; idx < buf1.size; idx++)
        ptr3[idx] = ptr1[idx] + ptr2[idx];
    return result;
}



PYBIND11_MODULE(example4, m) {
    m.doc() = "pybind11 example-1 plugin"; // optional module docstring


    m.def("add_c", &add_c, "A function which adds two arrays with c type");
}
