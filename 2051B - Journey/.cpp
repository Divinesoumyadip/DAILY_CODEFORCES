#include<bits/stdc++.h>
using namespace std;
int main(){
	int t;
	cin>>t;
	while(t--){
		int n,a,b,c;
		cin>>n>>a>>b>>c;
		cout<<n/(a+b+c)*3+(n%(a+b+c)>0)+(n%(a+b+c)>a)+(n%(a+b+c)>a+b)<<'\n';
	}
}
