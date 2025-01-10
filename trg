#include<bits/stdc++.h>
using namespace std;
#define int long long
const int N=5e5+10;
int t,n,rt,x,y,d[N],d2[N],a[N];
vector<int>e[N];
void dfs(int u,int fa){
	d[u]=d[fa]+a[u],d2[u]=d2[fa]+1;
	if(d[u]>d[rt]) rt=u;
	for(int v:e[u])
		if(v!=fa) dfs(v,u);
}
signed main(){
	cin>>t;
	while(t--){
		cin>>n; rt=0;
		for(int i=1;i<=n;++i) a[i]=0,e[i].clear();
		for(int i=1;i<n;++i) cin>>x>>y,++a[x],++a[y],e[x].push_back(y),e[y].push_back(x);
		for(int i=1;i<=n;++i) a[i]=(a[i]==1?1:a[i]-2);
		dfs(1,0),dfs(rt,0);
		cout<<d[rt]-(d2[rt]<3)-(d2[rt]<2)<<endl;
	}
	return 0;
}
