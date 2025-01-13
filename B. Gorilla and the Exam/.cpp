#include <iostream>
#include <vector>
using namespace std;
int T,n,k,cnt;
int main(){
	for(cin>>T;T--;){
		cin>>n>>k;cnt=n/k;
		for(int i=1;i<=n;++i){
			cout<<(i%k?++cnt:i/k)<<" \n"[i==n];}
	}
	return 0;
}
