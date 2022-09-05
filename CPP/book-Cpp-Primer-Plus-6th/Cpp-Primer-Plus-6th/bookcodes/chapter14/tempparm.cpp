#include <iostream>
#include "stacktp.h"

// 模板的参数是一个模板类
template <template <typename T> class Thing>
class Crab
{
private:
    Thing<int> s1;
    Thing<double> s2;
public:
    Crab() {};
    // assumes the thing class has push() and pop() members
    bool push(int a, double x) { return s1.push(a) && s2.push(x); }
    bool pop(int & a, double & x){ return s1.pop(a) && s2.pop(x); }
};

// 在模板中使用多种类型：类模板和其他类型定义
// 实例化的时候可以使用 myclass<Stack, int, double> instance;
template < template <typename T> class C, typename U, typename V>
class myclass{
    C<U> value1;
    C<V> value2;
};
    
int main()
{
    using std::cout;
    using std::cin;
    using std::endl;
    Crab<Stack> nebula;
// Stack must match template <typename T> class thing   
    int ni;
    double nb;
    cout << "Enter int double pairs, such as 4 3.5 (0 0 to end):\n";
    while (cin>> ni >> nb && ni > 0 && nb > 0)
    {
        if (!nebula.push(ni, nb))
            break;
    }
   
    while (nebula.pop(ni, nb))
           cout << ni << ", " << nb << endl;
    cout << "Done.\n";
    // cin.get();
    // cin.get();
    return 0; 
}
