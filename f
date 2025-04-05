#include <bits/stdc++.h>
using namespace std;
const int N = 200005;
struct edge{
    int a,b,c;
}g[N];
int n,m,tt,f[N],hd[N],vis[N],ans[N],to[N*2],nxt[N*2],id,tot,tl;
int find(int x) {return f[x]==x?x:f[x]=find(f[x]);}
void add(int a,int b) {to[++id]=b;nxt[id]=hd[a];hd[a]=id;}
bool dfs(int u,int fa) {
	ans[++tot]=u;
	if(u==tl) {
		printf("%d\n",tot);
		for(int i=1;i<=tot;i++) printf("%d ",ans[i]);
		cout<<endl;return 1;
	}vis[u]=1;
	for(int i=hd[u];i!=-1;i=nxt[i]) {
		if(to[i]!=fa&&vis[to[i]]==0&&dfs(to[i],u)) return 1;
	}tot--;return 0;
}
void work() {
	cin>>n>>m;int pos;
	for(int i=1;i<=m;i++)
		scanf("%d%d%d",&g[i].a,&g[i].b,&g[i].c);
	sort(g+1,g+m+1,[](edge a,edge b){return a.c>b.c;});
	for(int i=1;i<=n;i++) f[i]=i,hd[i]=-1,vis[i]=0;
	for(int i=1;i<=m;i++) {
		int ta=find(g[i].a),tb=find(g[i].b);
		if(ta==tb) pos=i;
		else f[ta]=tb;
	}printf("%d ",g[pos].c);id=0;
	for(int i=1;i<=m;i++) {
		add(g[i].a,g[i].b);
		add(g[i].b,g[i].a);
	}
	tl=g[pos].a;tot=0;
	dfs(g[pos].b,g[pos].a);
}
signed main() {cin>>tt;while(tt--) work();}
