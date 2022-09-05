// manyfrnd.cpp -- unbound template friend to a template class
#include <iostream>
using std::cout;
using std::endl;

typedef std::array<int, 12> arri;

// 使用typedef制定一系列别名
// C++11的特性
template<typename T>
    using arraytype = std::array< T, 12>;

arraytype<int> a;

template <typename T>
class ManyFriend
{
private:
    T item;
public:
    ManyFriend(const T & i) : item(i) {}
    // 是所有具体化的ManyFriend的友元函数，可以访问所有具体化的实例的成员变量
    template <typename C, typename D> friend void show2(C &, D &);
};

template <typename C, typename D> 
void show2(C & c, D & d)
{
    cout << c.item << ", " << d.item << endl;
}

int main()
{
    ManyFriend<int> hfi1(10);
    ManyFriend<int> hfi2(20);
    ManyFriend<double> hfdb(10.5);
    cout << "hfi1, hfi2: ";
    show2< ManyFriend<int>, ManyFriend<int> >(hfi1, hfi2);
    cout << "hfdb, hfi2: ";
    show2(hfdb, hfi2);
    // std::cin.get();
    return 0;
}
