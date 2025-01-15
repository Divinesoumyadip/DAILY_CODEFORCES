#include<bits/stdc++.h>
using namespace std;
int f[100010][3];
string s1,s2;
int count(int l1,int r1,int l2,int r2){
	int cnt=0;
	for(int i=l1;i<=r1;i++)cnt+=(s1[i]=='A');
	for(int i=l2;i<=r2;i++)cnt+=(s2[i]=='A');
	return cnt>1;
}
int main(){
	int t;
	cin>>t;
	while(t--){
		memset(f,0,sizeof(f));
		int n;
		cin>>n;
		cin>>s1>>s2;
		s1=" "+s1;
		s2=" "+s2;
		f[2][1]=count(1,2,1,1);
		f[1][2]=count(1,1,1,2);
		f[1][1]=f[2][2]=-1e9;
		for(int i=3;i<=n;i++){
			f[i][0]=max(max((f[i-1][1])+count(i,i,i-1,i),(f[i-2][2])+count(i-1,i,i,i)),
			(f[i-3][0]+count(i-2,i,i+1,i)+count(i+1,i,i-2,i)));
			f[i][1]=max((f[i-3][1]+count(i-2,i,i+1,i)+count(i+1,i,i-3,i-1)),(f[i-2][0]+count(i-1,i,i-1,i-1)));
			f[i][2]=max((f[i-3][2]+count(i-2,i,i+1,i)+count(i+1,i,i-1,i+1)),(f[i-1][0]+count(i,i,i,i+1)));
		}
		cout<<f[n][0]<<endl;
	}
	return 0;
}
