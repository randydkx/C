/**
 * @file using_derived.cpp
 * @brief 函数重装
 * @author 光城
 * @version v1
 * @date 2019-08-07
 */

#include <iostream>
using namespace std;

class Base{
    int base;
    public:
        void f(){ cout<<"f()"<<endl;
        }
        void f(int n){
            cout<<"Base::f(int)"<<endl;
        }
};

// 私有继承：Base中的元素和方法将会成为Derived类中的私有方法和私有成员变量
// 可以通过Base::f方式获取Base中的成员函数和变量，但是不能直接访问
// 私有继承相当于一个has-a的关系
class Derived : private Base {
    public:
    // 如果不将Base::f设置为public则会因为Base中的f函数设置为私有而不能访问
        using Base::f;
        void f(int n){
            // base = 10;
            cout<<"Derived::f(int)"<<endl;
        }
};

int main()
{
    Base b;
    Derived d;
    d.f();
    d.f(1);
    return 0;
}
