// 防止头文件的两次引用
/**
 * @author Wenshui Luo 
 * 
 */
#ifndef PrimerPlus_H_
#define PrimerPlus_H_

#include<iostream>
#include<vector>
#include<exception>
#include<map>
#include<typeinfo>
#include<string>
#include<fstream>
#include<functional>
#include<unordered_map>
using namespace std;

void function1(const int (*Listp)[3]){
    cout<<Listp<<endl;
    // Lisp中存放的是指向int*的指针，所以*Listp指向了int*的首地址
    cout << *(*(Listp + 1) + 1) << endl;
    // 第一个元素
    cout << **Listp<<endl;
    // 指向指针的指针对应的地址（二维指针的地址）
    cout<<"&Listp: "<<(&Listp)<<endl;
    cout<<"&*Listp: "<<&(*(Listp))<<" : "<<&(*(Listp+1))<<" : "<<&(*(Listp+2))<<endl;
    // 指针所指向的地址
    cout<<"Listp: "<<Listp<<"  Listp+1:"<<(Listp + 1)<<endl;
    cout<<"*Listp: "<<*Listp<<"  *(Listp+1):"<<*(Listp + 1)<<"  *(Listp+2):"<<*(Listp + 2)<<endl;
    return ;
}


void function2(int a){
    cout<< a <<endl;
}

void function3(int a){
    cout<< 2 * a <<endl;
}
// 使用函数指针进行参数传递，并且传递该函数需要的参数
void function4(void (*pf)(int), int a ){
    (*pf)(a);
}
// 声明一个函数，该函数接受一个int参数，同时返回int*类型的参数
// 调用： *(*p)(10)或者*p(10)
int* (*p)(int);
// 声明一个指向三个元素数组的指针pp，其中每个元素都是一个函数的指针，类型由定义的其他部分描述
void (*(* pp)[2])(int);
const double * d;
// 该函数声明将会返回一个结构的引用，如果返回int则返回一份拷贝
const pair<int,int>& function5(const int* &a, const int b);

struct node{
    int x;
    // 使用mutale 限定的类或结构体变量，使用const限定该结构体实例时也能修改该变量，但是其他变量不能修改
    mutable int y;

    node(){}
    node(int x,int y){this->x = x, this->y = y;}
    // 由于该函数隐式使用了this实例，用const限定当前的方法，使之不会修改使用的this实例
    void show() const
    {
        cout<<this->x<<" : "<<this->y<<endl;
    }
    // 表达类接口的两种方式：友元函数+类成员函数
    node operator+(const node & other) const{
        node tmp = node();
        tmp.x = this->x + other.x ;
        tmp.y = this->y + other.y;
        return tmp;
    }
    // 只能通过node * m调用
     node operator*(double m) const{
        node tmp = node();
        tmp.x = this->x * m;
        tmp.y = this->y * m;
        return tmp;
    }
    // 在类声明中使用友元函数，使得在外部定义的函数能够使用类的私有变量，同时解决了运算符的重载调用问题
    // 该函数虽然不是成员函数，但是和成员函数的访问权限一致
    // 能够通过m * node 调用
    friend node operator*(double m, const node& cur);
    // 使用友元函数重载输出运算符，使得可以连续输出
    friend ostream & operator << (ostream & os, const node& cur);
    // 转换函数，将类类型转换成int类型，必须是成员函数
    operator int() const;
    operator double() const;
    // 重载赋值运算符，因为C++生成的类赋值运算符是浅复制，比如将两个char* 内容指向同一块内存，析构则会产生问题
    // 将一个已有对象复制给另一个对象的时候会调用赋值运算符
    // 比如 :
    // node a = node();
    // node b = node();
    // a = b; 
    // 但是初始化的时候不一定会调用赋值运算符，一般调用复制构造函数
    // 最后的引用指向调用者自己，即*this
    // 只要在类中有需要通过new申请动态内存的部分（比如有一个属性是int * p)，就需要重新定义赋值运算符、复制构造函数和显式析构函数
    node& operator=(const node & other);
    // 移动赋值函数
    // node& operator=(node && other);
    // 拷贝构造函数（复制构造函数）,没有返回值，将另一个类拷贝到当前类，在如下的情境中触发
    // node * p = new node(other_node)
    // node a = node(other_node)
    // node a(other_node)
    // node a = b
    node(const node& other);
    // 移动构造函数，使用右值作为输入，并不会创建新的对象，而是将this对象指向右值对象在内存中的内容
    // 移动语义
    // node x,y;
    // node x = y将会使用复制构造函数
    // node z;
    // x = node(y+z)将会使用移动复制函数
    node(const node&& other){this->x = other.x,this->y=other.y;}

    // 重新定义()运算符从而能够实现类对象可以被调用，类似于python中的__call___(self,*)
    void operator()(){
        cout << "call node() as a function"<<endl;
    }
};

node & node::operator=(const node & other){
    if(&other == this)return *this;
    this->x = other.x;
    this->y = other.y;
    return *this;
}

node::node(const node & other){
    this->x = other.x, this->y = other.y;
}
// 友元函数在外部进行定义，这样可以改变运算的顺序，即
node operator*(double m, const node& other){
    return other * m;
}
ostream & operator << (ostream & os, const node& cur){
    os <<"content - "<< cur.x <<" : "<<cur.y;
    return os;
}
node::operator int() const{
    return this->x;
}

// 函数模板，但是没法对结构体做个性化的swap
template<typename T >
void function5(T &a, T &b){
    T tmp;
    tmp = a;
    a = b;
    b = tmp;
}

// 显式具体化，使得函数模板能够用该特定化函数处理node数据，下面两种是等价描述
template< > void function5<node> (node & a, node & b);
template< > void function5 (node & a, node & b);
// 显式实例化，编译器通过该函数模板生成int类型的function5，
// 告诉编译器不要使用function5来生成对于int的function5的功能定义，而是直接使用下面声明的函数
template void function5<int>(int & a, int & b);

// 使用函数模板声明多个类型，并自动判断表达式的结果类型
// 其中使用auto占用返回类型位置，是C++11的特性，其中->decltype(a + b)为后置返回类型，表示在函数中传参之后才会计算的类型
template< typename T1, typename T2>
auto function6(T1 a, T2 b) -> decltype(a + b){
    typedef decltype(a + b) abtype;
    abtype apb = a + b;
    abtype c[10];
    abtype & refa = c[2];
    return apb;
}


// ostream有cout所属类别，同时也能赋值ofstream对象
void file_it(ostream os, double fo, const double fe[], int n);

// 其中第一个const表示不能修改每个字符串的内容，即每个month[i]对应的字符串都是不可修改的，即const对应于char*的限定
// 第二个const表示不能改变month[i]的指向，比如不能用month[i] = "3"，即const是对应于month[i](一个指向char*数据的指针）的限定
const char * const month[2] = {"1","2"};

void function7(){
    // 函数中的静态局部变量，在内存中开辟固定位置，即是函数销毁该内存位置依然存在，该变量初始化一次，用作函数的闭包
    static int count = 0;
    cout << ++ count <<endl ;
}
// 未命名的名称空间相当于定义了static storage & internal linkage
// 等价于在其中定义了static int func;
namespace{
    int func;
}

struct TT{
    // 使用数组在拷贝赋值的时候将会对每个元素进行拷贝，
    // 如果使用node *p，并在构造函数中使用new，则会直接赋值指针指向的地址，即指针的值
    node p[2];
    TT(int x,int y,int x1, int y1){
        this->p[0] = node(x, y);
        this->p[1] = node(x1, y1);
    }
    // 使用virtual声明的类方法，在使用基类指针或者引用指向派生类对象时，如果调用派生类的方法，则会根据对象类型选择调用的方法
    // 该模式属于动态联编，dynamic binding
    // 每个对象中都会在编译时添加一个变量（指向虚函数表的指针），从而根据该指针确定对象应当调用的虚函数
    virtual void print(int n){
    // void print(int n){
        cout<< "TT "<<endl;
    }
};

struct TT2: TT{
    // 派生类中重定义的重名的方法将会覆盖基类中的方法，该行为不属于函数的重载，而属于函数的覆盖
    // 不管加不加virtual，对于基类来说都是一种覆盖（不可见）
    void print(){
        TT::print(10);
        cout<< "TT2 "<<endl;
    }
    TT2(TT &t1):TT(t1){}
};

// ABC抽象基类，因为其中至少包含一个纯虚函数
// 抽象基类无法生成对象，只有它的派生类可以产生对象
class BaseEllipse{
private:
    double x,y;
public:
    BaseEllipse(double x0 = 0, double y0 = 0): x(x0), y(y0){};
    virtual ~BaseEllipse(){}
    void Move(int nx, int ny){x = nx, y = ny;}
    // 纯虚函数，末尾一定加上=0,包含该函数的类一定是抽象基类
    virtual double Area() const = 0;
};

class Error{
public:
    void mesg();
};
void Error::mesg(){
    cout<<"a failure"<<endl;
}

void function8(){
    throw "a failure";
}

void function9(){
    throw Error();
}

// 使用函数包装器
template <typename T>
T use_f(T v, std::function< T(T)> f){
    cout<<"using function wrapper : "<<f(v)<<endl;
}

template<typename T>
void show_(T v){cout<<v;}
// 递归显示：
template<typename T,typename... Args>
void show_(T v, Args... args){
    cout<<v<<" , ";
    show_(args...);
}

int main(){
    int *p_list = new int[10];
    delete [] p_list;
    char s[10] = "123123";
    cout<<strlen(s)<<endl;

    int ar[3][3] = {{0,1,2},{3,4,5},{7,8,9}};
    function1(ar);
    
    function4(function2, 1);
    function4(function3, 1);

    // 隐含了对p占用的内存的初始化
    node * p = new node();
    // 列表初始化
    node p1 = {1,2};
    cout << p1 << endl;

    int a = 10,b = 11;
    function5(a, b);
    cout<< a<<" : "<<b<<endl;

    function7();
    function7();
    // 新分配内存的列表初始化
    int * arr = new int[4]{1,2,3,4};
    
    cout<< sizeof (*p)<<endl;

    // C++11中声明一个空指针
    char * str = nullptr;

    TT t1 = TT(1,2,3,4);
    TT t2 = TT(t1);
    cout<< "address : "<<(t1.p) <<" : "<<(t2.p)<<endl;
    cout<<"size : "<<sizeof t1<<endl;
    TT2 t3 = TT2(t1);
    t1.print(10);
    t3.print();

    // 使用try-catch捕获异常
    try{
        function8();
    }catch(const char * error){
        cout<<error<<endl;
    }

    // 第二种方式，返回一个类型，从而根据catch之后得到的类型捕获并处理
    try{
        function9();
    }catch(Error e){
        e.mesg();
    }

    // 获取某个实例所属的类型
    cout<<"type : "<<typeid( t2 ).name()<<endl;

    string wenshui = "input.txt";
    ofstream fout;
    char * strbuffer = new char[20];
    strcpy(strbuffer, wenshui.c_str());
    cout<< "copyed :" << strbuffer <<endl;
    fout.open(wenshui.c_str());
    cout << "status : "<<fout.is_open()<<endl;

    // 测试智能指针
    // 删除ap之后，ap的析构函数会自动释放ap指向的内存，从而自动回收内存
    // 所有的智能指针都有一个explicit构造函数，以及throw()设定，表示不会引发异常
    // auto_ptr<double> ap(new double);
    // *ap = 23.23;

    shared_ptr<string> sp(new string("this is  a shared pointer test"));
    cout<< *sp <<endl;

    unique_ptr<int> up(new int);
    *up = 10;
    cout<< *up<<endl;
    //使用所有权模式时，如果创建了临时对象unique pointer，则可以直接使用up指向，如果对象将存在一段时间，则赋值错误
    up = unique_ptr<int>(new int());

    vector<int> vec = {1,2,3};
    vector<int> vec2({1,3,4});
    vector<int> vec3 {1,2,3};

    // 无须map，使用hash实现，map是用树形结构存储的，插入和查找都是O(log)复杂度，
    unordered_map<string,int> m;

    // 右值引用，虽然不能获取10的地址，但是通过右值引用，将会将10存在内存中的位置
    // 可以通过&获取右值引用的地址
    int && aaa = 10;
    cout<< "address of a : "<< &aaa<<endl;

    node temp = node(1, 3);
    temp();
    // lambda表达式
    auto fun = [](int x){return x % 13 == 0;};
    // 中括号中的是当前作用域中的变量，加&则表示是该对象的引用
    // 使用[=]表示能够按值访问所有对象，使用[&]表示能够按照引用访问所有对象，
    // 使用[=, &temp]表示能够通过&访问temp，其他对象使用按值访问
    // 使用[&, vec]表示能够通过按值访问vec，其他对象都是按引用访问
    auto fun1 = [&temp](int x){temp();};
    auto fun2 = [=, &temp](int x){};
    auto fun3 = [&, temp](int x){};
    auto fun4 = [=](int x){};
    auto fun5 = [&](int x){};
    cout << fun(13) << endl;

    function<double(double)> fun6 = [](double x){return x * x;};
    function<double* (int* ,double, node)> f;

    use_f(10.0, fun6);

    show_('1',"3213",12);

    return 0;
}

#endif