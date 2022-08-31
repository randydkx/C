#include <iostream>
using namespace std;


int main()
{

    // int a = 12;
    // int b;
    // int *p;
    // int **ptr;
    // p = &a;     //&a的结果是一个指针，类型是int*，指向的类型是int，指向的地址是a的地址。
    // *p = 24;    //*p的结果，在这里它的类型是int，它所占用的地址是p所指向的地址，显然，*p就是变量a。
    // ptr = &p;   //&p的结果是个指针，该指针的类型是p的类型加个*，在这里是int**。该指针所指向的类型是p的类型，这里是int*。该指针所指向的地址就是指针p自己的地址。
    // *ptr = &b;  //*ptr是个指针，&b的结果也是个指针，且这两个指针的类型和所指向的类型是一样的，所以?amp;b来给*ptr赋值就是毫无问题的了。
    // **ptr = 34; //*ptr的结果是ptr所指向的东西，在这里是一个指针，对这个指针再做一次*运算，结果就是一个int类型的变量。
    // cout << (&(*ptr));

    // int arr[5] = { 0, 1, 2, 3, 4 };
	// //数组的指针，即指向int[5]的指针的指针
	// int (*arrPtr)[5] = &arr;
	// //指针的数组
	// int *ptrArr[5] = { &arr[0], &arr[1], &arr[2], &arr[3], &arr[4] };
    // cout << "arr: " << arr << endl;
    // cout << "&arr: " << &arr << endl;
    // cout << "&arrPtr: " << &arrPtr << endl;
	// cout << "arrPtr: " << arrPtr << endl;
	// cout << "*arrPtr:" << *arrPtr << endl;
	// for (int i = 0; i < 5;i++)
	// {
	// 	cout << ( *arrPtr ) [i] << " ";
	// 	cout << ptrArr[i] << " ";
	// 	cout << *(ptrArr[i] ) << " "<< endl;
	// }
    
    int *ps = new int;
    // int *pq = ps;
    delete ps;
    ps = new int;

    short tell[10];
    cout << tell << endl; //显示 &tell[0];
    cout << tell + 1 << endl;
    cout << &tell << endl; //显示整个数组的地址
    cout << &tell + 1 << endl;

    cout << "sizeof(ps) = " << sizeof(ps) << endl;

    int a[10][10];
    cout << &a[0] <<" : "<< &a[0] + 1<< " : " << a + 1 << endl;

    char chararr[5]={'A','B','C','D'};
    char (*p3)[5] = &chararr;

    int tmp[4]={1,2,3,4};
    int *ptr1=(int *)(&tmp+1);
    cout << ptr1[-1] << endl;

    char* ch[3] = {"123","456","789"};
    char ** chptr = ch;

    int *p[3];
    int arr[3][4];
    for(int i=0;i<3;i++)
        p[i]=a[i];

    int tmp1[3] = {1,2,3};
    int * p = tmp1;
}