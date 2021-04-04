#include <iostream>
#include <algorithm>
#include <string>
#include<vector>
#include<list>
#include<map>
#include<set>
#include<iterator>
#include<iomanip>
#include<unordered_map>
#include<queue> 
#include"test.h"

using namespace std;

void fun(int * a){
    a[0]=10;
}

struct node{
    int x,y;
    bool operator < (const node& a) const {
        return this->x < a.x;
    }
};
bool cmp(int& x,int& y){
    return x > y;
}


int main()
{
    vector<int> v;
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);
    v.push_back(4);

    list<int> l;
    list<int>::const_iterator i  ;
    cout<<INT_MAX<<endl;
    int *point = NULL;
    cout<<point<<endl;

    int a = 10;
    void* p1 = &a;
    cout<<*(int*)p1<<endl;

    char c[] = "12312";
    cout<<c<<endl;
    for(int i=0;i<26;i++){
        cout<<(char)(i+'a');
    }
    int s[]={2,4,4};
    fun(s);
    cout<<endl;
    cout<<s[0]<<endl;

    node node1,node2;
    node1.x = 1,node2.x = 2;
    cout<<(node1<node2)<<endl;

    hex(cout);
    cout<<12<<endl;

    oct(cout);
    cout<<12<<endl;
    
    dec(cout);
    cout<<12<<endl;

    cout<<setprecision(10)<<acos(-1L)<<endl;

    cout<<string::npos<<endl;

    string S = "1312";
    cout<<S.capacity()<<endl;
    cout<<S.size()<<endl;
    cout<<S.substr(0,2)<<endl;

    auto number = 10;
    cout<<number<<endl;

    vector<int> v3{1,3,4,3};

    for(auto i=v3.begin();i<v3.end();i++){
        cout<<*i<<endl;
    }
    cout<<v3.data()<<endl;
    reverse(v3.begin(),v3.end());
    for(int i=0;i<v3.size();i++){
        cout<<v3.at(i)<<endl;
    }

    cout<<"首元素"<<v3.front()<<";末尾元素："<<v3.back()<<endl;
    
    for (auto &x: v3){
        cout<<x<<endl;
    }

    v3.emplace_back(23);
    v3.push_back(12);
    v3.pop_back();

    v3.insert(v3.begin(),23);
    v3.insert(v3.end(),23);
    v3.insert(v3.begin()+2,2,87);
    v3.insert(v3.end(),{2,3,44,4});
    vector<int> v4 = {1,2,3};
    // 将v4中指定范围的元素插入v3的指定位置中
    v3.insert(v3.begin(),v4.begin(),v4.end());

    v3.erase(v3.begin()+1);
    v3.erase(v3.begin()+1,v3.end()-3);

    // 删除指定的元素,在algorithm头文件中，将v3 vector中元素23都删除
    remove(v3.begin(),v3.end(),23);
    // 删除v3中的所有元素
    v3.clear();
    cout<<"删除之后v3的大小"<<v3.size()<<endl;
    // 创建一个vector，其中元素都是23
    vector<int> v5(10,23);

    list<int> l2(v5.begin(),v5.end());

    // 将v3中的元素删除重复，将重复的元素删掉
    v3.erase(unique(v3.begin(),v3.end()),v3.end());

    
    map<string,int> mp;
    mp["string1"]=1;
    mp["string2"]=2;
    for(map<string,int>::iterator it=mp.begin();it != mp.end();it++){
        cout<<it->first<<" "<<it->second<<endl;
    }

    pair<int,int> p = make_pair(1,3);
    
    map<int,char> container{make_pair(1,'2')};
    // 删除指定的元素
    container.erase(1);
    cout<<container.size()<<endl;

    map<int,char>::iterator position = container.find(1);
    char news = 's';
    cout<<position->first<<endl;

    // 创建多重映射集，根节点是关键字最大的结点，其中的比较器按照关键字进行比较，所以设置为greater<string>
    multimap<string,int,greater<string> > mp2;

    mp2.insert(make_pair("123",23));
    mp2.insert(make_pair("123",43));
    mp2.insert(make_pair("345",234));

    for ( multimap<string,int>::iterator it = mp2.begin();it != mp2.end();it ++ ){
        cout<<it->first<<" "<<it->second<<endl;
    }
    // 查看其中键为123的数量
    cout << mp2.count("123") << endl;

    set<int> set1;
    set1.insert({1,2,34});
    // insert的返回值第一个表示的是插入数据的迭代器，第二个表示的是是否插入成功
    // 单重集合可能会插入失败（有相同的键）
    auto result =  set1.insert(234);
    
    if ( set1.insert(12).second ){
        cout << "成功插入一条数据" << endl;
    }

    set<int> set2 = {1,2,3,4,5,6,7,8,9};
    // 在set中删除元素之后it迭代器指向的位置表示的试下一个元素所在的位置
    // 删除所有的奇数
    for( set<int>::iterator it = set2.begin() ;it != set2.end() ;){
        if ( *it % 2 == 1)
           { 
                it = set2.erase(it);
                cout << "erase one element" << endl;
           }
        else 
            it ++ ;
    }
    cout << set2.size() << endl;

    // 使用greater实现最小堆
    priority_queue<int,vector<int>,greater<int> > q;
    q.push(1);
    q.push(-1);
    cout<<"首元素："<<q.top()<<endl;

    vector<int> v6 = {1,2,3,4,5};

    sort(v6.begin(),v6.end(),cmp);
    for (int i=0;i<v6.size();i++){
        cout<<v6[i]<<endl;
    }

    // 使用vector测试查找功能
    vector<char> forsearch={'1','2','3','4'};
    
    vector<char>::iterator it = find(forsearch.begin(),forsearch.end(),'1');
    if(it != forsearch.end()){
        cout << *it << endl;
    }
    // 排序，将最小的四个元素移到开头的位置并且将其排序，排序的复杂度是O(n*log(m)),m:头部有序的长度
    int to_sort[] = {9,8,7,6,5,4,3,2,1,0};
    vector<int> to_sort_vector;
    for(int i=0;i<10;i++){
        to_sort_vector.push_back(to_sort[i]);
    }
    partial_sort(to_sort_vector.begin(),to_sort_vector.begin()+4,to_sort_vector.end());
    for(auto &x : to_sort_vector){
        printf("%d ",x);
    }

    cout<<endl;

    // 使用nth_element将第n大的元素移动到第n的位置，保证第n个元素之前的比n小，之后得都比n大
    // 使用cmp比较器使得在n之前的更大
    nth_element(to_sort_vector.begin(),to_sort_vector.begin()+6,to_sort_vector.end(),cmp);
    for(auto x: to_sort_vector){
        cout<<x<<" ";
    }
    
}