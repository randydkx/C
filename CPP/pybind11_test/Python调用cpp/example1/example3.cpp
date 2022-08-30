/*****************************************************
@File    :   example3.cpp
@Time    :   2022/08/27 19:21:36
@Author  :   Wenshui Luo
@Email   :   randylo@njust.edu.com
@Copyright : None
*****************************************************/
#include<iostream>
#include <pybind11/pybind11.h>
namespace py = pybind11;

struct Pet_example3 {
    Pet_example3(const std::string &name) : name(name) { }
    std::string name;
};

struct Dog_example3 : Pet_example3 {
    Dog_example3(const std::string &name) : Pet_example3(name) { }
    std::string bark() const { return "woof!"; }
};

// 使用native类型，能够在python中通过复制内存的方式传递native类型
void print_vector(const std::vector<int> &v) {
    for (auto item : v)
        std::cout << item << " ";
}

// 为了在python中传递numpy的二维数组构建的缓冲区
class Gemfield {
public:
    Gemfield(size_t rows, size_t cols) : m_rows(rows), m_cols(cols) {
        m_data = new double[rows*cols];
    }
    double *data() { return m_data; }
    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }
private:
    size_t m_rows, m_cols;
    double *m_data;
};

static Gemfield gemfield(70,30);

Gemfield* getGem() {
    return &gemfield;
}

void test(Gemfield vi){
    std::cout<<vi.cols()<<" x "<<vi.rows()<<std::endl;
}

#include<pybind11/stl.h>
// 继承
PYBIND11_MODULE(example3, m){
    py::class_<Pet_example3>(m, "Pet_example3")
        .def(py::init<const std::string & >())
        .def_readwrite("name", &Pet_example3::name);
    
    py::class_<Dog_example3, Pet_example3>(m, "Dog_example3")
        .def(py::init<const std::string &>())
        .def("bark", &Dog_example3::bark);
        // dui error的支持
    m.def("error", [](){throw std::runtime_error("runtime error!");});
    m.def("print_vector", &print_vector);

    pybind11::class_<Gemfield>(m, "Gemfield", pybind11::buffer_protocol())
        .def(pybind11::init([](pybind11::buffer const b) {
            pybind11::buffer_info info = b.request();
            if (info.format != pybind11::format_descriptor<double>::format() || info.ndim != 2)
                throw std::runtime_error("Incompatible buffer format!");

            auto v = new Gemfield(info.shape[0], info.shape[1]);
            memcpy(v->data(), info.ptr, sizeof(double) * (size_t) (v->rows() * v->cols()));
            return v;
        }));

    m.def("test", &test);

    // 防止python将引用的内存认为是自己的从而将内存释放，c++再次释放时将造成core crash，故保证返回的是引用
    m.def("getgem", &getGem, py::return_value_policy::reference);
}