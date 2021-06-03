#include<stdlib.h>
#include<cstdio>
#include<time.h>
#include<iostream>
#include<sys/time.h>
#include<omp.h>
using namespace std;

double func(int n){
    return 1.0 / (0.1 / double(n) + 0.9 / (16 - double(n) * double(n)));
}
double func2(int n){
    return 1.0 / (0.04 / double(n) + 0.96 / (16 - double(n) * double(n)));
}

int main(int argc,char** argv){

    for(int i=1;i<=3;i++){
        cout<<"n="<<i<<" ,func="<<func(i)<<endl;
    }
    cout<<endl;
    for(int i=1;i<=3;i++){
        cout<<"n="<<i<<" ,func="<<func2(i)<<endl;
    }

    cout<<1.0/(0.04 + 0.96/15.0)<<endl;
    
    // long long int num_in_cycle=0;
    // long long int num_point;
    // int thread_count;
    // thread_count=1;
    // // scanf("%lld",&num_point);
    // num_point = 10000000;
    // srand(time(NULL));
    // double x,y,distance_point;
    // long long int i;
    // double start,stop;
    // struct timeval tv;
    // gettimeofday(&tv, NULL);
    // start = (tv.tv_sec) * 1000 + (tv.tv_usec) / 1000;
    // #pragma omp parallel for num_threads(thread_count) default(none) \
    //     reduction(+:num_in_cycle) shared(num_point) private(i,x,y,distance_point)
    // for( i=0;i<num_point;i++){
    //     x=(double)rand()/(double)RAND_MAX;
    //     y=(double)rand()/(double)RAND_MAX;
    //     distance_point=x*x+y*y;
    //     if(distance_point<=1){
    //         num_in_cycle++;
    //     }
    // }
    // double estimate_pi=(double)num_in_cycle/num_point*4;
    // printf("the estimate value of pi is %lf\n",estimate_pi);
    // gettimeofday(&tv, NULL);
    // stop = (tv.tv_sec) * 1000 + (tv.tv_usec) / 1000;
    // cout<<"total time is: "<<stop-start<<" ms"<<endl;
    // return 0;
}