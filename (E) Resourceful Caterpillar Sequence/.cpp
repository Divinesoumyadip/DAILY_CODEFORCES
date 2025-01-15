#include <bits/stdc++.h>
using namespace std;
const int MAXN=2e5+5;
vector<int>g[MAXN];
int n,notleaf[MAXN],sum,t;
long long ans;
signed main()
{
	cin>>t;
	while(t--)
	{
		sum=0;
		for(int i=1;i<=n;i++) g[i].clear(),notleaf[i]=0;
		cin>>n;
		for(int i=1,u,v;i<n;i++)
		{
			cin>>u>>v;
			g[u].push_back(v),g[v].push_back(u);
		}
		for(int i=1;i<=n;i++) sum+=(g[i].size()==1);
		ans=1ll*(n-sum)*sum;
		sum=0;
		for(int i=1;i<=n;i++)
		{
			if(g[i].size()!=1)
			{
				for(int v:g[i]) notleaf[i]+=(g[v].size()!=1);
				sum+=(notleaf[i]==g[i].size());
			}
		}
		for(int i=1;i<=n;i++)
		{
			if(g[i].size()!=1 && notleaf[i]!=g[i].size())
			{
				ans+=1ll*sum*(notleaf[i]-1);
			}
		}
		cout<<ans<<endl;
	}
}
