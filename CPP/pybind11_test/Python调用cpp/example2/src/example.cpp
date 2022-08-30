#include <iostream>
#include <vector>
#include "beincluded.h"

using std::vector;

// 向量内积，返回一个标量，输入的向量要等长
double inner_product(const vector<double>& vec1, const vector<double>& vec2){
    double sum = 0;
    for(int i = 0; i < vec1.size(); i++){
        sum += vec1[i] * vec2[i];
    }
    return sum;
}

// 向量求和，返回一个向量
vector<double> sum(const vector<double>& vec1, const vector<double>& vec2){
    auto result = vec1;
    for(int i = 0; i < vec1.size(); i++){
        result[i] += vec2[i];
    }
    return result;
}

// 定义一个向量类
class Vector{
public:
    Vector()=default;
    ~Vector()=default;

    //  以下重新实现了上述函数 
    double inner_product(const vector<double>& vec1, const vector<double>& vec2){
        print();
        double sum = 0;
        for(int i = 0; i < vec1.size(); i++){
            sum += vec1[i] * vec2[i];
        }
        return sum;
    }

    vector<double> sum(const vector<double>& vec1, const vector<double>& vec2){
        print();
        auto result = vec1;
        for(int i = 0; i < vec1.size(); i++){
            result[i] += vec2[i];
        }
        return result;
    }
};

// 使用pybind11将以上函数和类封装为python包
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // 函数中用到了C++的STL库，所以要包含该头文件
PYBIND11_MODULE(example, m){
    m.doc() = "my example";
    // 封装函数
    m.def("inner_product", &inner_product);
    m.def("sum", &sum);
    
    // 封装用lambda表达式表示的函数
    // 以下函数用于输出指定字符串
    m.def("print", [](std::string& str){
        std::cout << str << std::endl;      
    }); 

    // 封装类
    pybind11::class_<Vector>(m, "Vector")
            .def(pybind11::init())
            .def("inner_product", &Vector::inner_product)
            .def("sum", &Vector::sum);
}
