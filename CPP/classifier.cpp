#include<iostream>
using namespace std;
int qian0,qian1,hou0,hou1;
#define maxn 100010
struct node{
	int y,result;
};
bool cmp(node& a,node& b){
	return a.y<=b.y;
}
node a[maxn];
int n;
int main(){
	freopen("input.txt","r",stdin);
	cin>>n;
	for(int i=1;i<=n;i++){
		scanf("%d%d",&a[i].y,&a[i].result);
		if(a[i].result == 0){
			hou0+=1;
		}else{
			hou1+=1;
		}
	}
	
	sort(a+1,a+n+1,cmp);
	int ans = -1;
	int temp0=0,temp1=0;
	int final = -1;
	for(int i=1;i<=n;i++){		
		if(i!=n && a[i].y==a[i+1].y){
			if(a[i].result == 0){
			temp0+=1;
			}else {
				temp1+=1;
			}
			continue;
		}
		if(a[i].result == 0){
			temp0+=1;
			}else {
				temp1+=1;
			}
		if(qian0+hou1>=final){
			ans=a[i].y;
			final = qian0+hou1;
		}
		qian0+=temp0;
		hou1-=temp1;
		temp0 = temp1 = 0;
	}
	cout<<ans<<endl;
}