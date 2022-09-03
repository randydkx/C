#include<iostream>
#include<cstring>


using namespace std;
class Person{
public:
    typedef enum {
        BOY = 0, 
        GIRL 
    }SexType;
    Person(char *n, int a,SexType s){
        name=new char[strlen(n)+1];
        strcpy(name,n);
        age=a;
        sex=s;
    }
    // 使用const修饰的函数中的this指针是一个const Person * const 类型的指针
    // const表示不改变类的成员变量
    // 函数将会解析成get_age(const Person * const this)
    int get_age() const{
    
        return this->age; 
    }
    // 没有const修饰的成员函数，将会this类型为：Person * const，表示this指针不能指向其他Person对象
    // 但是可以改变this指针指向的对象（自身）的值
    // 函数将会被解析成 add_age(Person * const this)
    Person& add_age(int a){
        age+=a;
        return *this; 
    }
    ~Person(){
        delete [] name;
    }
private:
    char * name;
    int age;
    SexType sex;
};


int main(){
    Person p("zhangsan",20,Person::BOY); 
    cout<<p.get_age()<<endl;

    return 0;
}
