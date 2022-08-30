#include<Python.h> 
#include <iostream>
using namespace std;
 
int main()
{
    Py_Initialize(); //1、初始化python接口
    
    //初始化使用的变量
    PyObject* pModule = NULL;
    PyObject* pFunc = NULL;
    PyObject* pName = NULL;
    
    //2、初始化python系统文件路径，保证可以访问到 .py文件
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('../')");
    
    //3、调用python文件名。当前的测试python文件名是 myadd.py
    // 在使用这个函数的时候，只需要写文件的名称就可以了。不用写后缀。
    pModule = PyImport_ImportModule("myadd");
    
    //4、调用函数
    pFunc = PyObject_GetAttrString(pModule, "AdditionFc");
    
    //5、给python传参数
    // 函数调用的参数传递均是以元组的形式打包的,2表示参数个数
    // 如果AdditionFc中只有一个参数时，写1就可以了
    PyObject* pArgs = PyTuple_New(2);
 
    // 0：第一个参数，传入 int 类型的值 2
    PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", 2)); 
    // 1：第二个参数，传入 int 类型的值 4
    PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", 4)); 
    
    // 6、使用C++的python接口调用该函数
    // 对Python无参函数的调用直接将pArgs参数设置成为NULL即可
    PyObject* pReturn = PyEval_CallObject(pFunc, pArgs);
    
    // 7、接收python计算好的返回值
    int nResult;
    // i表示转换成int型变量。
    // 在这里，最需要注意的是：PyArg_Parse的最后一个参数，必须加上“&”符号
    PyArg_Parse(pReturn, "i", &nResult);
    cout << "return result is " << nResult << endl;
    
    //8、结束python接口初始化
    Py_Finalize();
}