#include<iostream>
#include<queue>
#include<vector>
#include<fstream>
#include<set>
using namespace std;

#define maxn 100
#define SEED 7
#define RANGE 100
#define NODECOUNT 50
const double THRESHOLD = 0.1;
set<int> allnodes = set<int>();
// 图中点和边的数量
int totalNodesCount = -1;
int totalEdgeCount = -1;
// 源点和漏点
int source=1,sink;
int currentIndex = 0;

struct OpenItem{
    int stateNode = -1;
    int fatherNode = -1;
    int cost = -1;
    OpenItem(){};
    OpenItem(int stateNode,int fatherNode,int cost){
        this->stateNode = stateNode;
        this->fatherNode = fatherNode;
        this->cost = cost;
    }
    bool operator < (const OpenItem& other) const {
        return this -> cost < other.cost;
    }
    void print(){
        cout<<"stateNode:"<<this->stateNode<<" fatherNode:"<<this->fatherNode<<" cost:"<<this->cost<<endl;
    }
};

struct CloseItem{
    int index = -1;
    int stateNode = -1;
    int fatherNode = -1;
    int cost = -1;
    CloseItem(){};
    CloseItem(int index,int stateNode,int fatherNode,int cost){
        this->index = index;
        this->stateNode = stateNode;
        this->fatherNode = fatherNode;
        this->cost = cost;
    }
    void print(){
        cout<<"index:"<<this->index;
        cout<<"stateNode:"<<this->stateNode<<" fatherNode:"<<this->fatherNode<<" cost:"<<this->cost<<endl;
    }
};

vector<OpenItem> open = vector<OpenItem>();
vector<CloseItem> close = vector<CloseItem>();

struct Edge{
    int to = -1;
    int weight = -1;
    Edge(){}
    Edge(int to,int weight){this->to = to,this->weight = weight;};
};

vector<Edge> nodes[maxn];

void GenerateGraph(int nodeCount){
    srand(SEED);
    string filename = "graph.txt";
    ofstream output(filename);
    for(int i=0;i<nodeCount;i++){
        for(int j=i+1;j<nodeCount;j++){
            double p = rand() / double(RAND_MAX);
            if(p < THRESHOLD){
                int weight = rand() % RANGE;
                if(output.is_open()){
                    output << i << " " << j << " " << weight << endl;
                }
                totalEdgeCount ++ ;
            }
        }
    }
    output.close();
}

//测试是否source和sink之间连通
bool testCanReach(int source ,int sink){
    bool vis[maxn];
    memset(vis,false,sizeof(vis));
    queue<int> queue;
    queue.push(source);
    vis[source]=true;
    while(!queue.empty()){
        int cur = queue.front();
        queue.pop();
        if(cur == sink)return true;
        vis[cur]=true;
        for (int i=0;i<nodes[cur].size();i++){
            int nex = nodes[cur][i].to;
            if(vis[nex])continue;
            vis[nex]=true;
            queue.push(nex);
        }
    }
    return false;
}

void initial(){
    // 初始化图数据结构
    for(int i=0;i<maxn;i++){
        nodes[i] = vector<Edge>();
    }
    while(!open.empty())open.pop_back();
    while(!close.empty())close.pop_back();
    GenerateGraph(NODECOUNT);
}

void printCur(int stage){
    cout<<"========================="<<stage<<"========================="<<endl;
    for(auto item : open){item.print();}
    cout<<endl;
    for(auto item : close){item.print();}
    cout<<endl;
}



int main(){
    freopen("graph.txt","r",stdin);
    int from,to,weight;
    initial();
    
    while((cin >> from) && from != EOF){
        cin >> to >> weight;
        allnodes.insert(from);
        allnodes.insert(to);
        nodes[from].push_back(Edge(to,weight));
        nodes[to].push_back(Edge(from,weight));
    }
    totalNodesCount = allnodes.size();
    source = 0,sink = totalNodesCount-1;
    while(!testCanReach(source,sink)){
        cout<<"test:"<<source << " " << sink<<endl;
        source = rand() % NODECOUNT;
        sink = rand() % NODECOUNT;
    }
    cout << "totalNodesCount: " << totalNodesCount << " totalEdgeCount: " << totalEdgeCount <<" source:" << source << " sink:" << sink << endl;
    for(int i=0;i<totalNodesCount;i++){
        for(int j=0;j<nodes[i].size();j++){
            cout << i << " " << nodes[i][j].to << " " << nodes[i][j].weight << " " << endl;
        }
    }
    open.push_back(OpenItem(source,-1,0));
    int stage = 0;
    bool hasAns = false;
    while(!open.empty()){
        // 将open表中的第一个元素取出来放到close表中
        OpenItem current = open.front();
        open.erase(open.begin());
        close.push_back(CloseItem(++currentIndex,current.stateNode,current.fatherNode,current.cost));
        // 是目标节点则退出
        if(current.stateNode == sink){
            hasAns = true;
            cout << "find an answer from " << source <<" to "<< sink << " : " << current.cost<< endl;
            break;
        }
        // 如果节点不可扩展则继续看open表
        if(!nodes[current.stateNode].size())continue;
        // 如果可以扩展则扩展
        for(int i=0;i<nodes[current.stateNode].size();i++){
            // 保证没有回边
            if(nodes[current.stateNode][i].to==current.stateNode)continue;
            open.push_back(OpenItem(nodes[current.stateNode][i].to,current.stateNode,nodes[current.stateNode][i].weight + current.cost));
        }
        // open表排序
        sort(open.begin(),open.end());
        // printCur(++stage);
    }
    if(!hasAns)cout<<"no answer"<<endl;
    return 0;
}