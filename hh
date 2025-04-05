#include <bits/stdc++.h>
using namespace std;

int main(){
    int n,x,y,a[51]={},b[51]={};
 cin>>n;
    for(int i=n*(n-1)/2-1;i;i--){
        cin>>x>>y;
        a[x]++;
        b[y]++;
    }
    for(int i=1;i<=n;i++) if(a[i]+b[i]!=n-1) x=y,y=i;
    if(a[x]<a[y]) swap(x,y);
   cout<<x<<" "<<y<<"\n";
}
