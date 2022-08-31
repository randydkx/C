// fowl.cpp  -- auto_ptr a poor choice
#include <iostream>
#include <string>
#include <memory>

int main()
{
    using namespace std;
    // shared pointers，仅仅当所指向的内容的指针计数减为0的时候将所指向的对象销毁
    shared_ptr<string> films[5] =
    {
        shared_ptr<string> (new string("Fowl Balls")),
        shared_ptr<string> (new string("Duck Walks")),
        shared_ptr<string> (new string("Chicken Runs")),
        shared_ptr<string> (new string("Turkey Errors")),
        shared_ptr<string> (new string("Goose Eggs"))
    };
    shared_ptr<string> pwin;
    pwin = films[2];   // films[2] loses ownership
    cout << "The nominees for best avian baseball film are\n";
    for (int i = 0; i < 5; i++){
        cout << "使用计数： " << films[i].use_count() << endl;
        cout << *films[i] << endl;}
    cout << "The winner is " << *pwin << "!\n";
    // cin.get();
    return 0;
}
