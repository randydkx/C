#include<iostream>
#include <omp.h>
#include<cstdlib>
#include<ctime>
using namespace std;
typedef long long ll;
const ll TIME = 1e6;
#define threads 10
int main() {

	clock_t start = clock();
	ll total = 0, i;
	double x, y;
	
	srand(time(NULL));
#pragma omp parallel for shared(TIME) reduction(+:total) private(x,y) num_threads(threads)
	for (i = 0; i < TIME; i++) {
		x = (double)rand() / (double)RAND_MAX;
		y = (double)rand() / (double)RAND_MAX;
		if (x * x + y * y <= 1.0)total++;
	}

	clock_t stop = clock();

	cout << (double)total / (double)TIME * 4.0 << endl;
	cout << "total time is: " << (double)(stop - start) / CLOCKS_PER_SEC * 1000 << " ms" << endl;

	return 0;
}

// #include<iostream>
// #include <omp.h>
// #include<sys/time.h>
// #include<ctime>
// using namespace std;
// typedef long long ll;
// const ll TIME = 1e7;
// #define threads 10
// int main(){
    
    
//     double start,stop;
//     struct timeval tv;
//     gettimeofday(&tv, NULL);
//     start = (tv.tv_sec) * 1000 + (tv.tv_usec) / 1000;
//     ll total = 0,i;
//     double x,y;
//     clock_t begin = clock();
//     srand(time(NULL));
//     #pragma omp parallel for reduction(+:total) private(x,y) num_threads(threads)
//         for(i=0;i<TIME;i++){    
//             x = (double)rand()/(double)RAND_MAX;
//             y = (double)rand()/(double)RAND_MAX;
//             if( x*x + y*y <= 1.0)total++;
//             // printf("%d\n",omp_get_thread_num()); 
//         }

//     gettimeofday(&tv, NULL);
//     stop = (tv.tv_sec) * 1000 + (tv.tv_usec) / 1000;
//     clock_t end = clock();

//     cout<<(double)total / (double)TIME * (double)4<<endl;
//     cout<<"total time is: "<<(end-begin)/CLOCKS_PER_SEC * 1000 <<" ms"<<endl;

//     return 0;
// }


// #include<iostream>
// #include<omp.h>
// #include<ctime>
// #include <cstdlib>

// #define MAX_ITERATION 1000000
// #define PRECISION 32767
// #define f(i,a,b) for(int i=a;i<=b;i++)
// #define ll long long
// #define Rand(a,b) (rand()%(b-a+1)+a)	//2^15=32768
// #define dbg(args) cout<<#args<<" : "<<(args)<<endl;

// using namespace std;
// //int main(int argc, char* argv[]) {
// int main() {	//调试用
// 	clock_t startTime, endTime;
// 	startTime = clock();

// 	// srand(time(NULL));
// 	// int test = -1;
// 	// f(i, 1, 50) {
// 	// 	test = max(test, rand());
// 	// 	//cout << rand()<<endl;
// 	// }
// 	// dbg(test);

// 	double x, y;
// 	int effective_counter = 0;

// // # pragma omp parallel for reduction(+:effective_counter) private(x,y)
// 	f(i, 1, MAX_ITERATION) {
// 		x = Rand(0, PRECISION);
// 		y = Rand(0, PRECISION);
// 		x /= PRECISION;
// 		y /= PRECISION;
// 		if (x * x + y * y <= 1) {
// 			effective_counter++;
// 		}
        
// 	}
// 	double PI = 4.0 * effective_counter / MAX_ITERATION;
// 	dbg(PI);

// 	endTime = clock();
// 	double tot_time = (double)(endTime - startTime) / CLOCKS_PER_SEC * 1000;
// 	cout << "The run time is: " << tot_time << " milliseconds" << endl;
// }