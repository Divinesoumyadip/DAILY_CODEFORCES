#include<bits/stdc++.h>
using namespace std;
const int N=2e5+5;
int a[N];
int st[N][31];
int main(){
	int t;
	cin>>t;
	while(t--){
		int n,q;
		cin>>n>>q;
		for(int i=0;i<n;i++){
			cin>>a[i];
			if(i) st[i][0]=a[i]-a[i-1];
		} 
		for(int j=1;j<20;j++)
	   for(int i=1;i<=n-(1<<j)+1;i++){
	   	st[i][j]=__gcd(st[i][j-1],st[i+(1<<(j-1))][j-1]);
	   }
		while(q--){
			int l,r;
			cin>>l>>r;
			r--;
			if(l>r) cout<<0<<" ";
			else{
				int len=log2(r-l+1);
				cout<<abs(__gcd(st[l][len],st[r-(1<<len)+1][len]))<<" ";
			}
			cout<<endl;
		}
	
	}
	return 0;
} 
