/**
 * @file test.cpp
 * @author @randydkx
 * @brief 测试程序
 * @version 0.1
 * @date 2021-04-04
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include<iostream>
#include<cstdio>
#include<list>
#include<iterator>
using namespace std;

int main(){
    
    list<int> l{1,2,3,4,5};
    std::list<int>::iterator it = l.begin();
    it ++ ;
    it = l.erase(it);

    return 0;
}
