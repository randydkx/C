/**
 * Key ideas:
 *
 *   - The task-based approach is typically superior to its thread-based counterpart.
 *
 *   - With the task-based approach, it is easy to get access to the return value of doAsyncWork,
 *     because the future returned from std::async offers the get function.
 *
 *   - The get function also provides access to the exception if doAsyncWork throws an exception.
 */
#include <future>
#include <iostream>

int doAsyncWork()
{
    std::cout << "doAsyncWork()" << std::endl;

    //throw;

    return 1;
}


int main()
{
    // 传递给std::async的函数对象表示一个任务，fut表示future，是一个任务定义
    auto fut = std::async(doAsyncWork);  // onus of thread mgmt is
                                         // on implementer of
					 // the Standard Library

    // 使用基于任务的线程调用可以获得调用之后的返回值，而不是thread那样的join()之后没有返回值和执行情况
    // 同时std::thread中没有异常
    int ret = fut.get();
    std::cout << "doAsyncWork() returned " << ret << std::endl;

    return 0;
}
