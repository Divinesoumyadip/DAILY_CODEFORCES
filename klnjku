#include<iostream>
#include<cassert>
using namespace std;
int K,M;
string S;
int main()
{
	ios::sync_with_stdio(false);
	cin.tie(nullptr);
	int T;cin>>T;
	for(;T--;)
	{
		cin>>K>>M;
		cin>>S;
		int pc=0;
		for(char c:S)if(c=='1')pc++;
		int ans=0;
		for(int m=1;m<=pc;m++)
		{
			int emp=(m+1)/2;
			//H(emp, pc-m) = C(emp-1+pc-m, emp-1)
			if((emp-1+pc-m&emp-1)==emp-1)
			{
				const int need=m-1;
				for(int k=0;k<30;k++)if(M>>k&1)
				{
					if(need>>k&1)continue;
					if((M>>k+1&need>>k+1)!=need>>k+1)continue;
					int t=k-__builtin_popcount(need&(1<<k)-1);
					if(t==0)ans^=M>>k+1<<k+1|(1<<k)-1;
					if(t==1)ans^=~need&(1<<k)-1;
				}
			}
		}
		cout<<ans<<"\n";
	}
}
