#include<bits/stdc++.h>
using namespace std;
const long long L=1e10;
inline bool check(string s){
	if(!isdigit(s[0])) return 0;
	if(!isdigit(s.back())) return 0;
	for(int i=1;i<s.length();i++){
		if(s[i-1]=='0'){
			if(isdigit(s[i])&&(i==1||!isdigit(s[i-2]))) return 0;
		}
		else if(!isdigit(s[i-1])){
			if(!isdigit(s[i])) return 0;
		}
	}
	long long x=0,cl=0,cr=0;
	int sgn=1;
	bool f=0;
	for(char i:s){
		if(isdigit(i)) x=x*10+i-'0';
		else{
			if(x>=L) return 0;
			f?cr+=sgn*x:cl+=sgn*x;
			if(i=='=') f=1;
			x=0,sgn=i=='-'?-1:1;
		}
	}
	if(x>=L) return 0;
	return cl==cr+sgn*x;
}
int main(){
	ios::sync_with_stdio(0);
	cin.tie(0);cout.tie(0);
	string s;cin>>s;
	if(check(s)) cout<<"Correct\n",exit(0);
	for(int i=0;i<s.length();i++)
		if(isdigit(s[i])){
			string tl=s,tr=s;
			for(int j=i;j;j--){
				swap(tl[j],tl[j-1]);
				if(check(tl))cout<<tl<<endl,exit(0);
			}
			for(int j=i;j<s.length()-1;j++){
				swap(tr[j],tr[j+1]);
				if(check(tr))cout<<tr<<endl,exit(0);
			}
		}
	cout<<"Impossible\n";
	return 0;
}
