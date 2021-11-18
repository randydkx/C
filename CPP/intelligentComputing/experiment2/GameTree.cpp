#include<iostream>
#include<queue>
#include<vector>
#include<fstream>
#include<set>
#include<cassert>
using namespace std;

// 棋盘宽度
#define WIDTH 5
// 对于任意一方的搜索深度
#define DEPTH_OF_SEARCH 4
// 无穷数
#define INFINITY 0x00ffffff

#ifndef DEBUG
    // #define DEBUG
#endif

// max或者min方的节点（即用于该方进行扩展）
enum NodeType{MIN_AND=0x01,MAX_OR=0x02};
// 棋子，分别是o,x,空
enum Type{MIN=0,MAX=1,empty};

class Chess{
    public:
        // 棋盘布局
        int chess[WIDTH][WIDTH];
    
    public:
        // 初始化棋局
        Chess();
        // 复制棋局
        void copy(Chess *chess);
        // 打印棋局
        void print();
        // 检查两个棋盘是否一致(不考虑翻转概念下的相等),即增加了对不同摆放下判断是否相等
        bool isEqualWithPosition(Chess chess);
        // 检查对应位置是否相等，不经过反射和旋转变换
        bool check(Chess chess);
        // 获取某个位置的值
        int getij(int i,int j);
        // 设置某个位置的值
        void setij(int i,int j,int value);
        // 查找MAX或者MIN下子的行、列、对角可覆盖的情况
        int valueBySymbol(int sym);
        // MAX或者Min必胜,sym=Type::MIN or Type::MAX
        bool canMaxOrMinWin(int sym);
        // 旋转90度棋局
        Chess rotate90();
        // 翻转棋盘,左右翻转
        Chess flip_lr();
        // 翻转棋盘，上下翻转
        Chess flip_ud();
        // 对角翻转，左上到右下为对称轴
        Chess flip_lu_rd();
        // 对角翻转，左下到右上
        Chess flip_ld_ru();
};

// 博弈树节点
class Node{
    public:
        int alpha,beta;
        int Evalue=-INFINITY;
        int depth=0;
        int type;//节点类型，or & and
        Chess chess;
        Node *father = NULL;//父节点
        Node *Next;//下一个转移的节点
        // Node对应的落子位置
        pair<int,int> p;
    public:
        Node(){
            this->chess = Chess();
            // alpha只增不减，所以设置最小值，只有或节点使用alpha
            this->alpha=-INFINITY;
            // beta只减不增，所以设置最大值，只有与节点使用beta
            this->beta=INFINITY;
        }
        int getEvalue(Type type);
        // 复制节点，只复制棋局
        static void copy(Node &a,Node b){
            a.chess.copy(&(b.chess));
        }
};

vector<Chess> atDepth[DEPTH_OF_SEARCH + 1];

Chess Chess::rotate90(){
    Chess ret ;
    for(int i=0;i<WIDTH;i++){
        for(int j=0;j<WIDTH;j++){
            ret.chess[j][WIDTH - 1 - i] = this->chess[i][j];
        }
    }
    return ret;
}

Chess Chess::flip_ud(){
    Chess ret;
    for(int i=0;i<WIDTH;i++){
        for(int j=0;j<WIDTH;j++){
            ret.chess[WIDTH - 1 - i][j] = this->chess[i][j];
        }
    }
    return ret;
}

Chess Chess::flip_lr(){
    Chess ret;
    for(int i=0;i<WIDTH;i++){
        for(int j=0;j<WIDTH;j++){
            ret.chess[i][WIDTH - 1 - j] = this->chess[i][j];
        }
    }
    return ret;
}

Chess Chess::flip_lu_rd(){
    Chess ret;
    for(int i=0;i<WIDTH;i++){
        for(int j=0;j<WIDTH;j++){
            ret.chess[j][i] = this->chess[i][j];
        }
    }
    return ret;
}

Chess Chess::flip_ld_ru(){
    Chess ret;
    for(int i=0;i<WIDTH;i++){
        for(int j=0;j<WIDTH;j++){
            ret.chess[WIDTH - 1 - j][WIDTH - 1 - i] = this->chess[i][j];
        }
    }
    return ret;
}

bool Chess::canMaxOrMinWin(int sym){
    bool flag = false;
    // 行列
    for(int i=0;i<WIDTH;i++){
        int count = 0;
        for(int k=0;k<5;k++)
            if(this->chess[i][k]==sym)count++;
        if(count==5)flag = true;
        count = 0;
        for(int k=0;k<5;k++)
            if(this->chess[k][i]==sym)count++;
        if(count==5)flag = true;
    }
    int count = 0;
    // 对角线
    for(int k=0;k<5;k++)
        if(this->chess[k][k]==sym)count++;
    if(count==5)flag = true;
    count = 0;
    for(int k=0;k<5;k++)
        if(this->chess[4-k][k]==sym)count++;
    if(count==5)flag = true;

    return flag;
}

int Chess::valueBySymbol(int sym){
    int maxEvalue = 0;
    // 行列
    for(int i=0;i<WIDTH;i++){
        int count = 0;
        for(int k=0;k<5;k++)
            if(this->chess[i][k]==sym || this->chess[i][k]==empty)count++;
        if(count == 5)maxEvalue++;
        count = 0;
        for(int k=0;k<5;k++)
            if(this->chess[k][i]==sym || this->chess[k][i]==empty)count++;
        if(count == 5)maxEvalue++;
    }

    // 对角线
    int count = 0;
    for(int k=0;k<5;k++)
        if(this->chess[k][k]==sym|| this->chess[k][k]==empty)count++;
    if(count == 5)maxEvalue++;
    count = 0;
    for(int k=0;k<5;k++)
        if(this->chess[4-k][k]==sym || this->chess[4-k][k]==empty)count++;
    if(count == 5)maxEvalue++;
    // cout<<maxEvalue<<endl;
    return maxEvalue;
}

Chess::Chess(){
    for(int i=0;i<WIDTH;i++)
        for(int j=0;j<WIDTH;j++)
            this->chess[i][j]=Type::empty;
}

void Chess::copy(Chess *other){
    for(int i=0;i<WIDTH;i++)
        for(int j=0;j<WIDTH;j++)
            this->chess[i][j]=other->chess[i][j];
}

void Chess::setij(int i,int j,int value){
    assert (i>=0 && i<=WIDTH - 1);
    assert (j>=0 && j<=WIDTH - 1);
    this->chess[i][j] = value;
}

int Chess::getij(int i,int j){
    assert (i>=0 && i<=WIDTH - 1);
    assert (j>=0 && j<=WIDTH - 1);
    return this->chess[i][j];
}

void Chess::print(){
    for(int i=0;i<WIDTH;i++){
        for(int j=0;j<WIDTH;j++){
            if(this->chess[i][j]==Type::MAX)
                cout<<"x ";
            else if(this->chess[i][j]==Type::MIN)
                cout<<"o ";
            else cout<<"_ ";
        }
        cout<<endl;
    }
    cout<<endl;
}

bool Chess::isEqualWithPosition(Chess other){
    bool flag = false;
    // 将当前棋局与另一个棋局的原棋局、旋转90、180、270的棋局进行比较
    for(int i=0;i<4;i++){
        if(this->check(other))flag=true;
        other = other.rotate90();
    }
    return flag;
}

bool Chess::check(Chess other){
    for(int i=0;i<WIDTH;i++)
        for(int j=0;j<WIDTH;j++)
            if(this->chess[i][j] != other.chess[i][j])return false;
    return true;
}

int Node::getEvalue(Type type){
    // MAX方
    if(type == Type::MAX){
        if(this->chess.canMaxOrMinWin(Type::MAX))return INFINITY;
        else if(this->chess.canMaxOrMinWin(Type::MIN))return -INFINITY;
        else return this->chess.valueBySymbol(Type::MAX) - this->chess.valueBySymbol(Type::MIN);
    }else{
        if(this->chess.canMaxOrMinWin(Type::MIN))return INFINITY;
        else if(this->chess.canMaxOrMinWin(Type::MAX))return -INFINITY;
        else return this->chess.valueBySymbol(Type::MIN) - this->chess.valueBySymbol(Type::MAX);
    }
}

// 将nextNode对应的空间在内存中复制一份，给current节点的Next节点所指向
void copy(Node& current,Node &nextNode){
    current.Next = new Node();
    current.Next->chess = nextNode.chess;
    current.Next->alpha = nextNode.alpha;
    current.Next->beta = nextNode.beta;
    current.Next->father = nextNode.father;
    current.Next->Evalue = nextNode.Evalue;
    current.Next->Next= nextNode.Next;
    current.Next->depth = nextNode.depth;
    current.Next->p = nextNode.p;
}

// 判断在当前深度是否搜索过同样的棋局，减少重复搜索的分支
bool isSearched(vector<Chess> atDepthk,Chess chess){
    for(int i=0;i<atDepthk.size();i++){
        Chess toCompare = atDepthk[i];
        // 不翻转，从四个位置对比，相等
        if(toCompare.isEqualWithPosition(chess))return true;
        // 上下或者左右翻转，从四个位置对比，相等
        else if(toCompare.flip_lr().isEqualWithPosition(chess))return true;
        else if(toCompare.flip_ud().isEqualWithPosition(chess))return true;
    }

    return false;
}

void clearDepth(){
    for(int i=0;i<=DEPTH_OF_SEARCH;i++){
        atDepth[i].clear();
    }
}

// 深搜，current:当前扩展节点，father:当前节点的父节点，type:下子类型，ox中的一种，用Type代替
// judge:是否对棋局判重
// 返回值，当前节点的估值
int DFS(Node &current,Node father,int& count,Type type,bool judge){
    for(int i=0;i<WIDTH;i++){
        for(int j=0;j<WIDTH;j++){
            if(current.chess.chess[i][j]==Type::empty){
                Node nextNode = current;
                nextNode.depth = current.depth + 1;
                if(nextNode.depth > DEPTH_OF_SEARCH)continue;
                nextNode.father = &current;
                nextNode.p.first = i;
                nextNode.p.second = j;
                if(current.type == NodeType::MAX_OR){
                    nextNode.chess.chess[i][j] = type;
                    nextNode.type = NodeType::MIN_AND;
                }else if(current.type == NodeType::MIN_AND){
                    nextNode.chess.chess[i][j] = 1-type;
                    nextNode.type = NodeType::MAX_OR;
                }
                // 叶子结点，直接计算Evalue,否则将由子节点的返回值计算
                if(nextNode.depth == DEPTH_OF_SEARCH)
                    nextNode.Evalue = nextNode.getEvalue(type);
                else{
                    // 与节点保证Evalue是上界，因为需要子节点最小的Evalue作为估值
                    // 或节点保证Evalue是下界，因为需要子节点中最大的Evalue作为估值，保证存在可以指向的子节点
                    if(nextNode.type == NodeType::MAX_OR)nextNode.Evalue = -INFINITY;
                    else nextNode.Evalue = INFINITY;
                }
                // 已经搜索过该棋局则放弃
                if(judge){
                    if(!isSearched(atDepth[nextNode.depth],nextNode.chess))
                        atDepth[nextNode.depth].push_back(nextNode.chess);
                    else continue;
                }

                #ifdef DEBUG
                cout<<"depth: "<<nextNode.depth<<"  Evalue: "<<nextNode.Evalue<<endl;
                nextNode.chess.print();
                #endif
                
                // 搜索子节点，返回子节点nextNode的估值，通过该估值改变当前节点的相关属性，搜索次数+=1
                count++;
                int ret = DFS(nextNode,current,count,type,judge);
                // 是一个或节点，则取子节点中估价最大的一个值作为扩展，并且更新alpha值
                if(current.type == NodeType::MAX_OR && (ret > current.Evalue)){
                    current.Evalue = ret;
                    current.alpha = max(ret,current.alpha);
                    copy(current,nextNode);
                    if(current.father != NULL && current.alpha >= current.father->beta)
                        return current.Evalue;

                }else if(current.type == NodeType::MIN_AND && (ret < current.Evalue)){
                    current.Evalue = ret;
                    // AND节点的倒退值是子节点中最小的一个
                    current.beta = min(ret,current.beta);
                    copy(current,nextNode);
                    // 剪枝，当前节点估计值的上确界beta<=父节点下确界alpha，不会产生更优的决策，递归在MIN节点返回
                    if(current.father != NULL && current.beta <= current.father->alpha)
                        return current.Evalue;
                        
                }
            }
        }
    }

    #ifdef DEBUG
    // 递归回溯的时候当前深度为1节点的Evalue已经确定，
    if(current.depth == 1){
        cout<<"after search: Evalue:"<<current.Evalue<<endl;
        current.chess.print();
    }
    #endif
    return current.Evalue;
}

// 打印一次博弈中的最佳深度搜索路径，长度为DEPTH+1
void printBestInOneGame(Node current){
    cout<<"============================="<<endl;
    for(int i=0;i<=DEPTH_OF_SEARCH;i++){
        // if(i!=1){current = *(current.Next);continue;}
        if(i & 1)cout<<"depth"<<i<<": 与节点"<<endl;
        else cout<<"depth"<<i<<": 或节点"<<endl;
        if(i != 0)cout<<"落子位置： ("<<current.p.first<<","<<current.p.second<<")"<<endl;
        current.chess.print();
        if(current.Next != NULL)
            current = *(current.Next);
        else break;
    }
}

void run(){
    Node node = Node();
    node.Evalue = -INFINITY;
    node.type = NodeType::MAX_OR;
    node.depth = 0;
    // 用来测试必胜局面
    // node.chess.setij(0,1,Type::MAX);
    // node.chess.setij(0,2,Type::MAX);
    // node.chess.setij(0,3,Type::MAX);
    // node.chess.setij(0,4,Type::MAX);
    // node.chess.setij(0,0,Type::MAX);
    // node.chess.setij(1,1,Type::MIN);
    node.chess.print();
    node.father = new Node();
    // 统计DFS搜索的次数
    int count = 0;

    // 进行一定次数的双方博弈测试
    bool flag = true;
    vector<Chess> Path=vector<Chess>();

    // 是否对棋局判重，判重则搜索深度最多4，不判重深度可以很高
    bool judge = false;
    // 初始棋局
    Path.push_back(node.chess);
    for(int i=0;i<20;i++){
        // 任何一方开始最优博弈之前都将搜索深度上的已搜索棋局清除
        clearDepth();
        // 先手，max方flag=true
        if(flag){
            // 任何一方都是用最优策略，将当前自己将要扩展的节点作为MAX节点
            count = 0;
            Node current = Node();
            Node::copy(current,node);
            current.type = NodeType::MAX_OR;
            current.depth = 0;
            current.Evalue = -INFINITY;
            current.father = new Node();
            DFS(current,*(current.father),count,Type::MAX,judge);
            cout<<"Step "<<i+1<<": MAX ———— 搜索次数："<<count<<endl;
            cout<<"Evalue: "<<current.Evalue<<endl;
            if(current.Evalue == INFINITY){
                cout<<"先手MAX获胜"<<endl;
                break;
            }else if(current.Evalue == -INFINITY){
                cout<<"先手MAX必败"<<endl;
                break;
            }
            assert (current.Next != NULL);
            // 打印最优博弈树中的最优路径
            printBestInOneGame(current);
            assert(current.Next->depth == 1);
            Node::copy(node,*(current.Next));
        }else{
            count = 0;
            Node current = Node();
            Node::copy(current,node);
            // MIN方搜索的时候也是使用同样的最优博弈
            current.type = NodeType::MAX_OR;
            current.depth = 0;
            current.Evalue = -INFINITY;
            current.father = new Node();
            DFS(current,*(current.father),count,Type::MIN,judge);
            cout<<"Step "<<i+1<<": MIN: ———— 搜索次数："<<count<<endl;
            cout<<"Evalue: "<<current.Evalue<<endl;
            if(current.Evalue == INFINITY){
                cout<<"先手MIN获胜"<<endl;
                break;
            }else if(current.Evalue == -INFINITY){
                cout<<"先手MIN必败"<<endl;
                break;
            }
            assert (current.Next != NULL);
            // 打印最优博弈树中的最优路径
            printBestInOneGame(current);
            assert(current.Next->depth == 1);
            Node::copy(node,*(current.Next));
        }
        flag  = !flag;    
    }
}

// void printDepth(){
//     for(int i=0;i<=DEPTH_OF_SEARCH;i++){
//         cout<<"深度:  "<<i<<endl;
//         for(Chess c : atDepth[i]){
//             c.print();
//         }
//     }
// }

void test2(){
    Node node = Node();
    node.Evalue = -INFINITY;
    node.type = NodeType::MAX_OR;
    node.father = new Node();
    node.depth = 0;
    node.chess.setij(0,1,Type::MIN);
    node.chess.setij(0,2,Type::MIN);
    node.chess.setij(0,3,Type::MIN);
    node.chess.setij(0,4,Type::MIN);
    // node.chess.setij(0,0,Type::MIN);
    node.chess.setij(1,1,Type::MAX);
    int count = 0;
    // 测试MAX方能够发现对方的胜局
    DFS(node,*(node.father),count,Type::MAX,true);
    printBestInOneGame(node);
    cout<<"==========================="<<endl;

    // 测试不同的变换
    node.chess.print();
    node.chess.flip_lr().print();
    node.chess.flip_ud().print();
    node.chess.rotate90().print();
    node.chess.flip_lu_rd().print();
    node.chess.flip_ld_ru().print();

    cout<<node.chess.isEqualWithPosition(node.chess.rotate90().rotate90().rotate90())<<endl;
}


int main(){
    freopen("output.txt","w",stdout);
    run();
    // printDepth();
    // test2();
    return 0;
}