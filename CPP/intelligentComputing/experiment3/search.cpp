#include<iostream>
#include<queue>
#include<vector>
#include<fstream>
#include<map>
#include<math.h>
#include<set>
#include<cassert>
using namespace std;

#define SEED 103
#define MAXN 50
#define RAND_OBSTACLES 50
#define PII pair<int,int>
#define MAX_SPEED 60
#define convert(a) (a * MAX_SPEED)
typedef vector<vector<double>> matrix;
const string file_name = "graph.txt";
int graph[MAXN][MAXN];
// 论域：分别为曲率、距离、车速
const double partitionRatio[] = {0.1,0.2,0.3,0.4,0.5};
const double partitionDis[] = {1,2,3,3.8,4};
const double partitionSpeed[] = {0.2,0.4,0.6,0.8,1.0};
PII source = make_pair(0,0);
PII sink = make_pair(MAXN - 1,MAXN - 1);

class node{
    public:
        int x,y;
        double currentCost;//当前的准确估价
        double totalCost;//估价
    public:
        node(int x,int y){this->x = x,this->y = y,this->currentCost = this-> totalCost = 0;}
        node(){this->x = -1,this->y = -1;}
        bool operator > (const node& other) const{
            return this->totalCost > other.totalCost;
        }
};
// 障碍物类
class obstacle{
    public:
        int x,y,width,height;
    public:
        obstacle(int x,int y,int height,int width){
            this->width = width;
            this->height = height;
            this->x=x;
            this->y=y;
        }
        obstacle(){};
};

vector<obstacle> obstacles;

PII father[MAXN][MAXN];

namespace Generator{
    void add(int x1,int y1,int height,int width){
        obstacles.push_back(obstacle(x1,y1,height,width));
        for(int i=x1;i < x1+height;i++){
            for(int j=y1;j < y1+width;j++){
                graph[i][j] = 1;
            }
        }
    }

    void save(string file_name){
        ofstream output(file_name);
        assert(output.is_open());
        for(int i=0;i<MAXN;i++){
            for(int j=0;j<MAXN;j++){
                output<<graph[i][j];
            }
            output<<endl;
        }
        output.close();
    }

    void run(){
        memset(graph,0x00,sizeof(graph));
        add(10,10,3,3);
        add(15,15,4,4);
        add(5,15,3,4);
        add(35,40,5,5);
        add(20,5,5,5);
        add(28,20,3,3);
        add(2,6,3,3);
        add(2,40,5,3);
        add(40,2,6,7);
        add(20,40,4,7);
        add(12,35,4,7);
        add(22,26,3,3);
        add(35,26,5,8);
        // 随机障碍
        for(int i=0;i<RAND_OBSTACLES;i++){
            int x = rand() % MAXN;
            int y = rand() % MAXN;
            add(x,y,1,1);
        }
        save(file_name);
    }
};

void loadGraph(){
    ifstream input(file_name);
    for(int i=0;i<MAXN;i++){
        string line;
        input >> line;
        for(int j=0;j<MAXN;j++){
            graph[i][j] = int(line[j]-'0');
        }
    }
}

void printGraph(int graph[][MAXN]){
    for(int i=0;i<MAXN;i++){
        for(int j=0;j<MAXN;j++){
            if(graph[i][j] == 0)cout<<" ";
            else if(graph[i][j] == 1)cout<<"x";
            else cout<<"o";
        }
        cout<<"|"<<endl;
    }
}

void printMatrix(matrix m){
    for(int i=0;i<m.size();i++){
        for(int j=0;j<m[0].size();j++){
            cout<<m[i][j]<<" "; 
        }
        cout<<endl;
    }
}

// 两点之间的欧式距离
double distance(PII a,PII b){
    PII v = make_pair(a.first - b.first,a.second - b.second);
    return sqrt(v.first*v.first + v.second*v.second);
}

void AStar(){
    // 创建最小堆
    int direction[][2] = {{-1,0},{0,1},{1,0},{0,-1},{-1,-1},{-1,1},{1,-1},{1,1}};
    priority_queue< node,vector<node>,greater<node> > heap;
    node start = node(source.first,source.second);
    start.currentCost = 0;
    // 使用曼哈顿距离估价
    start.totalCost = abs(sink.first - start.x) + abs(sink.second - start.y);
    heap.push(start);
    bool vis[MAXN][MAXN];
    memset(vis,false,sizeof(vis));
    vis[start.x][start.y]=true;
    while(!heap.empty()){
        node current = heap.top();
        heap.pop();
        // 搜索8个方向
        for(int i=0;i<8;i++){
            node next = current;
            next.x = current.x + direction[i][0];
            next.y = current.y + direction[i][1];
            if(next.x < 0 || next.x >= MAXN || next.y < 0 || next.y >= MAXN)continue;
            if(graph[next.x][next.y])continue;
            if(vis[next.x][next.y])continue;
            vis[next.x][next.y] = true;
            father[next.x][next.y] = {current.x,current.y};
            // 搜索到目标节点
            if(next.x == sink.first && next.y == sink.second){
                cout<<"find an answer"<<endl;
                return;
            }
            next.currentCost += distance({next.x,next.y},{current.x,current.y});
            // 预测估价，使用曼哈顿距离
            double h = abs(sink.first - next.x) + abs(sink.second - next.y);
            next.totalCost = next.currentCost + h;
            
            heap.push(next);
        }
    }
}

vector<PII> printPath(){
    int currentx = sink.first,currenty = sink.second;
    stack<PII> stack;
    stack.push({currentx,currenty});
    while(father[currentx][currenty].first!=-1){
        int xx = father[currentx][currenty].first;
        int yy = father[currentx][currenty].second;
        assert(xx >=0 && xx < MAXN && yy >=0 && yy < MAXN);
        stack.push({xx,yy});
        currentx = xx;
        currenty = yy;
    }
    vector<PII> path;
    while(!stack.empty()){
        path.push_back(stack.top());
        stack.pop();
    }
    int g[MAXN][MAXN];
    for(int i=0;i<MAXN;i++)
        for(int j=0;j<MAXN;j++)g[i][j] = graph[i][j];
    for(int i=0;i<path.size();i++)
        g[path[i].first][path[i].second] = 2;
    printGraph(g);
    return path;
}



// 使用余弦函数衡量曲率，也可以通过曲率公式计算，
// 结果:一个[0,1]区间的实数，表示曲率大小，值越大曲率越大
double Ratio(PII a,PII b,PII c){
    PII vector1 = make_pair(a.first - b.first,a.second - b.second);
    PII vector2 = make_pair(c.first - b.first,c.second - b.second);
    double product = (double)vector1.first * vector2.first + vector1.second * vector2.second;
    double size1 = distance(a,b);
    double size2 = distance(c,b);
    double cosine = product / (size1 * size2);
    return (cosine + 1) / 2;//[0,1]
}

double singleObstacleDistance(PII position,obstacle o){
    double dis = 10000000.0;
    int x = position.first,y = position.second;
    // 由于图比较小，所以直接暴力计算两个点之间的距离，取最小的作为道障碍物之间的距离
    for(int i=o.x;i<o.x+o.height;i++){
        for(int j=o.y;j<o.y + o.width;j++){
            dis = min(dis,distance({x,y},{i,j}));
        }
    }
    return dis;
}

// 到障碍物之间最短的距离，作为到障碍物的距离度量
double distanceToObstacles(PII position,vector<obstacle> os){
    double distance = 100000000.0;
    for(int i=0;i<os.size();i++){
        obstacle cur = os.at(i);
        distance = min(singleObstacleDistance(position,cur),distance);
    }
    return distance;
}   

matrix getVagueMatrix(double *a,double *b,int na,int nb){
    matrix ret;;
    for(int i=0;i<na;i++){
        ret.push_back(vector<double>());
        for(int j=0;j<nb;j++){
            ret[i].push_back(min(a[i],b[j]));
        }
    }
    return ret;
}

vector<double> getRWithWeight(matrix a,matrix b,double lambda,vector<double> v1,vector<double> v2){
    //m1:曲率小对应的论域大小，m2:距离远对应的论域大小，n:车速快对应的论域数量
    int m1 = a.size(),m2 = b.size(),n = a.at(0).size();
    vector<double> ans1 = vector<double>(),ans2 = vector<double>(),ans;
    for(int i=0;i<n;i++){
        double tmp = .0;
        for(int j=0;j<m1;j++){
            tmp = max(tmp,min(a[i][j],v1[i]));
        }
        ans1.push_back(tmp);
    }
    for(int i=0;i<n;i++){
        double tmp = .0;
        for(int j=0;j<m2;j++){
            tmp = max(tmp,min(a[i][j],v2[i]));
        }
        ans2.push_back(tmp);
    }
    double sumAns1 = .0,sumAns2 = .0;
    // 对隶属度向量进行归一化处理
    for(int i=0;i<n;i++){
        sumAns1 += ans1.at(i);
        sumAns2 += ans2.at(i);
    }
    for(int i=0;i<n;i++)ans1.at(i) /= sumAns1,ans2.at(i) /= sumAns2;
    
    assert(ans1.size() == ans2.size());
    for(int i=0;i<ans1.size();i++)
        ans.push_back(ans1.at(i)*lambda + ans2.at(i)*(1 - lambda));

    return ans;
}

// 计算曲率和距离对应的论域元素，可以认为是阶
void getLevel(double ratio,double dis,vector<double> &ratioVec,vector<double> &disVec){
    int index1 = 0,index2 = 0;
    
    for(int i=0;i<5;i++){
        if( (i==0 && ratio < partitionRatio[i]) || (i != 4 && ratio < partitionRatio[i] && ratio >= partitionRatio[i - 1])
        || (i == 4 && (ratio >= partitionRatio[i]))){
            index1 = i;
            break;
        }
    }
    for(int i=0;i<5;i++){
        if( (i==0 && dis < partitionDis[i]) || (i != 4 && dis < partitionDis[i] && dis >= partitionDis[i - 1])
        || (i == 4 && (dis >= partitionDis[i]))){
            index2 = i;
            break;
        }
    }
    // 将所属的档位标注为0.8，表示极大可能隶属于该档位
    // 其他档位设置为均匀值
    ratioVec[index1] = 0.8;
    disVec[index2] = 0.8;
    for(int i=0;i<5;i++){
        if(i != index1)ratioVec[i] = (1.0 - 0.8) / 4;
        if(i != index2)disVec[i] = (1.0 - 0.8) / 4;
    }
}

// 使用最大决策或者加权决策
double getSpeedFromDecisionVector(vector<double> decisionVec,string type){
    if(type == "max"){
        int speedRatioIndex = 0;
        double maximum = .0;
        for(int i=0;i<decisionVec.size();i++){
            if(decisionVec.at(i) > maximum){
                speedRatioIndex = i;
                maximum = decisionVec.at(i);
            }
        }
        return partitionSpeed[speedRatioIndex];
    }else{
        double a = .0,b = .0;
        for(int i=0;i<decisionVec.size();i++){
            a += decisionVec[i] * partitionSpeed[i];
            b += decisionVec[i];
        }
        return a / b;
    }
}


// 模糊推理，为path上的每个点计算车速，设置车速最大为max，则范围为[0,max]
vector<double> VagueInference(vector<PII> path,vector<obstacle> obs,vector<double> &distanceList,vector<double> &ratioList){
    // 曲率小的隶属度,以及曲率小的分段阶数
    double miuOfLowRatio[] = {0.1,0.2,0.3,0.4,0.5};
    int numOfLowRatio = 5;
    // 车速快的隶属度
    double miuOfHighSpeed[] = {0.1,0.2,0.6,0.8,1.0};
    int numOfHighSpeed = 5;
    // 距离远的隶属度
    double miuOfLongDis[] = {0.2,0.4,0.6,0.8,1.0};
    int numOfLogDis = 5;
    // 分别计算曲率小-车速快的模糊矩阵 和 距离远-车速快的模糊矩阵
    matrix R1 = getVagueMatrix(miuOfLowRatio,miuOfHighSpeed,numOfLowRatio,numOfHighSpeed);
    cout<<"曲率小-车速快的模糊矩阵："<<endl;
    printMatrix(R1);
    matrix R2 = getVagueMatrix(miuOfLongDis,miuOfHighSpeed,numOfLogDis,numOfHighSpeed);
    cout<<"距离远-车速快的模糊矩阵："<<endl;
    printMatrix(R2);
    // 合成最终模糊矩阵，加权表示两者的重要性，以映射到[0,1]区间中
    // lambda表示曲率的重要性
    double lambda = 0.5;
    vector<double> speedList = vector<double>();
    for(int i=0;i<=path.size()-3;i++){
        // 计算该位置下对应的曲率隶属向量和距离隶属向量
        vector<double> vectorOfRatio,vectorOfDis;
        for(int j=0;j<numOfLowRatio;j++)vectorOfRatio.push_back(.0);
        for(int j=0;j<numOfLogDis;j++)vectorOfDis.push_back(.0);
        double ratioValue = Ratio(path[i],path[i + 1],path[i + 2]);
        ratioList.push_back(ratioValue);
        double disValue = distanceToObstacles(path[i],obstacles);
        distanceList.push_back(disValue);
        getLevel(ratioValue,disValue,vectorOfRatio,vectorOfDis);
        vector<double> decisionVec = getRWithWeight(R1,R2,lambda,vectorOfRatio,vectorOfDis);
        speedList.push_back(getSpeedFromDecisionVector(decisionVec,"max"));
    }

    return speedList;
}

void initial(){
    for(int i=0;i<MAXN;i++){
        for(int j=0;j<MAXN;j++){
            father[i][j] = make_pair(-1,-1);
        }
    }
}


int main(){
    // freopen("output_100x100_RAND_OBSTACLES_2000.txt","w",stdout);
    initial();
    // 建图与保存
    Generator::run();
    // 导入图
    loadGraph();
    // A*算法构建最短路
    AStar();
    // 打印路径图
    vector<PII> path = printPath();
    vector<double> distanceList = vector<double>();
    vector<double> ratioList = vector<double>();
    // 模糊推理计算路径
    vector<double> speedList = VagueInference(path,obstacles,distanceList,ratioList);
    
    for(int i=0;i<=path.size()-3;i++){
        PII cur = path.at(i);
        cout<<i<<" : point("<<cur.first<<","<<cur.second<<")"<<"\tdistance:"<< distanceList[i]<<"\tratio:"<<ratioList[i]<<"\tspeed:"<<convert(speedList.at(i))<<endl;
    }
}