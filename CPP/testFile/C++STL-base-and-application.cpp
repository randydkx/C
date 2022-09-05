/**
 * @file C++STL-base-and-application.cpp
 * @author wenshuiluo
 * @brief 关于c++STL的测试
 * @version 0.1
 * @date 2021-04-04
 * 
 * @copyright Copyright (c) 2021
 * 
 */


#include<iostream>
// 字符串输入输出流
#include<sstream>
#include<string>
#include<vector>
using namespace std;

int main(){
    // 文件重定向
    // freopen("input.txt","r",stdin);
    // freopen("output.txt","w",stdout);

    /**
     * @brief 字符串输入输出流
     * 
     */
    int n;
    float f;
    string strhello;
    string strtext = "1 2.31 hello";
    istringstream s(strtext);
    s >> n;
    s >> f;
    s >> strhello;
    cout << "字符流输出： " << n << " " << f << " " << strhello << endl;

    string str = "hello";

    // 需要include<vector>
    std::vector<int> intv_{1,2,3,4};
    auto pos = find(intv_.cbegin(),intv_.cend(),2);
    intv_.insert(pos, -1);
    for(const auto & ele : intv_){
        cout << "intv_ (modified): " << ele << std::endl;
    }
    cout << "=========" << std::endl;
    auto pos_2 = find(intv_.rbegin(), intv_.rend(), 2) - intv_.rbegin();
    cout << pos_2 << std::endl;
    for(const auto & ele : intv_){
        cout << "intv_ (modified): " << ele << std::endl;
    }
    int x{0};
    /**
     * @brief 获取string的长度
     * 
     */
    cout << "字符串的长度：" << str.length() << endl;
    // 在指定位置插入数据
    string toinsert = "llll";
    str.insert(1,toinsert);
    cout << "插入数据之后的string:" << str << endl;

    /**
     * @brief 字符串替换操作,从0位置开始，删除2个字符，然后插入第三个参数中的string
     * 
     */
    str.replace(0,2,"test");
    cout << "替换之后的string：" << str << endl;

    /**
     * @brief 字符串查找
     * 
     */
    string s1 = "what's your name?my name is TOM, HOwdo you do ? Fine , thanks.";
    cout << "your 第一个出现的位置 " << s1.find("your") << endl;    
    // 从指定位置开始查找，如果查找失败将会返回string::npos
    cout << "查找失败时find的返回值：" << s1.find("your",16) << endl;
    assert(s1.find("your",16) == string::npos);
    cout << "查找成功时string的返回值： " << s1.find_first_of("your") << endl;

    /**
     * @brief 字符串删除
     * 
     */
    string s2 = "123456";
    s2.erase(s2.begin(),s2.begin()+2);
    cout << "删除开头两个字符" << s2 << endl;
    s2 = "123456";
    cout << "另一种方式删除开头两个字符" << s2.erase(0,2) << endl;
    s2 = "1 2 3 4 5 6";
    s2.erase(find(s2.begin(),s2.end(),' '));
    cout << "删除空格： " << s2 << endl;
    

    /**
     * @brief 添加和删除末尾的字符
     * 
     */

    s2.push_back('a');
    cout << "push_back之后： " << s2 <<endl;
    s2.pop_back();
    cout << "pop_back之后：" << s2 << endl;

    /**
     * @brief 获取string的子串
     * 
     */
    cout << "从0开始长度为10的子串" << s2.substr(0,6) << endl;

    /**
     * @brief 交换字符串
     * 
     */
    string string1 = "1234";
    string string2 = "5678";
    cout << "交换之前: " << string1 << " " << string2 << endl;
    string1.swap(string2);
    cout << "交换之后： " << string1 << " " << string2 << endl;
    
    int int1 = 10, int2 = 20;
    swap( int1 ,int2 );
    cout << "交换之后的int值：" << int1 << " " << int2 <<endl;

    
}