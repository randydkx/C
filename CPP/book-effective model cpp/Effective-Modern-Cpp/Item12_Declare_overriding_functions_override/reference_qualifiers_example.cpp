/*
 * Key idea:
 *
 *   Member function reference qualifiers are one of C++11's less-publicized
 *   features and make it possible to limit use of a member function to lvalues
 *   only or to rvalues only.
 */

#include <iostream>

class Widget {
public:

// 左值引用对象调用该函数时的行为
  void doWork() & {                          // this version of doWork applies only
    std::cout << "doWork() &" << std::endl;  // when *this is an lvalue
  }

// 右值引用对象调用该函数时的行为
  void doWork() && {                         // this version of doWork applies only
    std::cout << "doWork() &&" << std::endl; // when *this is an rvalue
  }

};

Widget makeWidget()      // factory function (returns rvalue)
{
  Widget w;
  return w;
}

int main()
{
  Widget w;               // normal object (an lvalue)

  // 调用者是一个左值
  w.doWork();             // calls Widget::doWork for lvalues
                          // (i.e., Widget::doWork &)

// 临时对象，调用右值引用doWork()函数
  makeWidget().doWork();  // calls Widget::doWork for rvalues
                          // (i.e., Widget::doWork &&)
}
