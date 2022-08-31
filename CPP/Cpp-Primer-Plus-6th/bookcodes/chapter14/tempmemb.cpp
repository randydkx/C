// tempmemb.cpp -- template members
#include <iostream>
using std::cout;
using std::endl;

template <typename T>
class beta
{
private:
    template <typename V>  // nested template class member
    class hold
    {
    private:
        V val;
    public:
        hold(V v  = 0) : val(v) {}
        void show() const { cout << val << endl; }
        V Value() const { return val; }
    };
    hold<T> q;             // template object
    hold<int> n;           // template object
public:
    beta( T t, int i) : q(t), n(i) {}
    template<typename U>   // template method
    U blab(U u, T t) { return (n.Value() + q.Value()) * u / t; }
    void Show() const { q.show(); n.show();}
};

template <typename T>
class beta2{
private:
    template <typename V>
    class inbeta;
public:
    template <typename U>
    // U show(U num){std::cout << num << std::endl; return U;}
    U show(U num);
};

// 在类外定义类中声明的private class，模板嵌套，标注生成的是一个class
template <typename T>
    template <typename V>
        class beta2<T>::inbeta{
            
        };

template <typename T>
    template <typename U>
        U beta2<T>::show(U num){}
        
int main()
{
    beta<double> guy(3.5, 3);
    cout << "T was set to double\n";
    guy.Show();
    cout << "V was set to T, which is double, then V was set to int\n";
    cout << guy.blab(10, 2.3) << endl;
    cout << "U was set to int\n";
    cout << guy.blab(10.0, 2.3) << endl;
    cout << "U was set to double\n";
    cout << "Done\n";
    // std::cin.get();
    return 0; 
}
