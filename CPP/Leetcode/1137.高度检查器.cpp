/*
 * @lc app=leetcode.cn id=1236 lang=cpp
 *
 * [1137] 第 N 个泰波那契数
 */

// @lc code=start
class Solution {
public:
    int tribonacci(int n) {
        int T[100];
        T[0]=0;
        T[1]=T[2]=1;
        for (int i = 3; i < n + 1; i++)
        {
            T[i] = T[i-1]+T[i-2]+T[i-3]
        }
        
        return T[n];
    }
};
// @lc code=end

