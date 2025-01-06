#include <bits/stdc++.h>
using namespace std;
int main() {
    long long d;
    string b,c,s="";
    cin >> b>>c;
    map<char,int> m;
    for(int i=1;i<b.length();i++){
    	if(m.count(b[i])==0)
    	m[b[i]]=i;
	}
	int y=INT_MAX;
	for(int j=1;j<c.length();j++){
		char x=c[c.length()-j-1];
		if(m.count(x)!=0&&j+m[x]<y){
			s=b.substr(0,m[x]+1)+c.substr(c.length()-j);
			y=j+m[x];
		}
	}
	if(s=="")
	cout<<-1;
	else
	cout<<s;
}
