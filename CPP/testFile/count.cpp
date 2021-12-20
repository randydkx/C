#include<iostream>
#include <stdlib.h>
#include<vector>
#include<cstdlib>
#include<queue>
#include<ctime>
using namespace std;

#define MAX_IN_ONE_DAY 10
#define TOTAL 1000

vector<int> get_random_squence(){
    vector<int> ret = vector<int>();
    for(int i=0;i<6;i++){
        int _rand = rand() % 6;
        ret.push_back(_rand);
    }
    return ret;
}

long policy[3][3][3][3][3][3][3] = {0};
#define p policy[i][j][k][l][m][n][t]

void print(vector<int> ret){
    for(int i=0;i<6;i++){
            cout<<ret.at(i)<<" ";
        }
        cout<<endl;
}

vector<int> total[7][MAX_IN_ONE_DAY];

int score_for_A(vector<int> seq){
    int a[6] = {0};
    for(auto ele : seq){a[ele]++;}
    return a[0] * a[1] * a[2];
}

int score_for_B(vector<int> seq,int& current_sel){
    int a[6] = {0};
    for(auto ele : seq){a[ele]++;}
    current_sel++;
    return a[3] * a[4] + current_sel;
}

int score_for_C(vector<int> seq,int& flag){
    int count_6 = 0;
    for(auto ele : seq){if(ele == 5)count_6++;}
    flag += count_6;
    if(flag >= 10){
        flag -= 10;
        return 100;
    }else{
        return 0;
    }
}

int main()
{
    srand(time(NULL));
    int ca = TOTAL;
    while(ca --){
        for(int day = 0;day < 7;day++){
            for(int i=0;i<MAX_IN_ONE_DAY;i++){
                total[day][i] = get_random_squence();
                // print(total[day][i]);
            }
        }
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                for(int k=0;k<3;k++){
                    for(int l=0;l<3;l++){
                        for(int m=0;m<3;m++){
                            for(int n=0;n<3;n++){
                                for(int t=0;t<3;t++){
                                    // 针对一个策略
                                    int sel_B = 0;
                                    int flag = 0;
                                    int score_total = 0;
                                    for(int day=0;day<7;day++){
                                        int max_s = 0;
                                        for(int resume=0;resume<MAX_IN_ONE_DAY;resume ++ ){
                                            
                                            vector<int> seq = total[day][resume];
                                            if(day==0){
                                                if(i == 0){
                                                    int score = score_for_A(seq);
                                                    max_s = max(score,max_s);
                                                }else if(i == 1){
                                                    int score = score_for_B(seq,sel_B);
                                                    max_s = max(score,max_s);
                                                }else if(i == 2){
                                                    int score = score_for_C(seq,flag);
                                                    max_s = max(score,max_s);
                                                }
                                            }else if(day == 1){
                                                if(j == 0){
                                                    int score = score_for_A(seq);
                                                    max_s = max(score,max_s);
                                                }else if(j == 1){
                                                    int score = score_for_B(seq,sel_B);
                                                    max_s = max(score,max_s);
                                                }else if(j == 2){
                                                    int score = score_for_C(seq,flag);
                                                    max_s = max(score,max_s);
                                                }
                                            }else if(day == 2){
                                                if(k == 0){
                                                    int score = score_for_A(seq);
                                                    max_s = max(score,max_s);
                                                }else if(k == 1){
                                                    int score = score_for_B(seq,sel_B);
                                                    max_s = max(score,max_s);
                                                }else if(k == 2){
                                                    int score = score_for_C(seq,flag);
                                                    max_s = max(score,max_s);
                                                }
                                            }else if(day == 3){
                                                if(l == 0){
                                                    int score = score_for_A(seq);
                                                    max_s = max(score,max_s);
                                                }else if(l == 1){
                                                    int score = score_for_B(seq,sel_B);
                                                    max_s = max(score,max_s);
                                                }else if(l == 2){
                                                    int score = score_for_C(seq,flag);
                                                    max_s = max(score,max_s);
                                                }
                                            }else if(day == 4){
                                                if(m == 0){
                                                    int score = score_for_A(seq);
                                                    max_s = max(score,max_s);
                                                }else if(m == 1){
                                                    int score = score_for_B(seq,sel_B);
                                                    max_s = max(score,max_s);
                                                }else if(m == 2){
                                                    int score = score_for_C(seq,flag);
                                                    max_s = max(score,max_s);
                                                }
                                            }else if(day == 5){
                                                if(n == 0){
                                                    int score = score_for_A(seq);
                                                    max_s = max(score,max_s);
                                                }else if(n == 1){
                                                    int score = score_for_B(seq,sel_B);
                                                    max_s = max(score,max_s);
                                                }else if(n == 2){
                                                    int score = score_for_C(seq,flag);
                                                    max_s = max(score,max_s);
                                                }
                                            }else if(day == 6){
                                                if(t == 0){
                                                    int score = score_for_A(seq);
                                                    max_s = max(score,max_s);
                                                }else if(t == 1){
                                                    int score = score_for_B(seq,sel_B);
                                                    max_s = max(score,max_s);
                                                }else if(t == 2){
                                                    int score = score_for_C(seq,flag);
                                                    max_s = max(score,max_s);
                                                }
                                            }
                                            // cout<<day<<" : "<<max_s<<endl;
                                        }
                                        score_total += max_s;
                                    }

                                    p += score_total;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    double max_score = 0.;
    vector<int> final_policy = vector<int>();
    for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                for(int k=0;k<3;k++){
                    for(int l=0;l<3;l++){
                        for(int m=0;m<3;m++){
                            for(int n=0;n<3;n++){
                                for(int t=0;t<3;t++){
                                    double score = (double) p / (double) TOTAL;
                                    cout<<i<<" "<<j<<" "<<k<<" "<<l<<" "<<m<<" "<<n<<" "<<t<<" -----> "<<score<<endl;
                                    if(score > max_score){
                                        max_score = score;
                                        final_policy.clear();
                                        final_policy.push_back(i);
                                        final_policy.push_back(j);
                                        final_policy.push_back(k);
                                        final_policy.push_back(l);
                                        final_policy.push_back(m);
                                        final_policy.push_back(n);
                                        final_policy.push_back(t);
                                    }
                                }
                            }
                        }
                    }
                }
            }
    }

    cout<<max_score<<endl;
    for(auto ele: final_policy){cout<<ele<<" ";}

}