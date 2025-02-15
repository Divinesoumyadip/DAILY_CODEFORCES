#include <bits/stdc++.h>
using namespace std;
void solve() {
	int n; cin >> n;
	string s, t; cin >> s >> t;
	int z = 0, o = 0;
	for(auto i: s) (i&1?o:z)++;
	for(auto i: t) {
		if(!o||!z) {cout<<"nO\n"; return;}
		(i&1?z:o)--;
	}
	cout << "yEs\n";
}
main() {
	int t; cin >> t;
	while(t--) solve();
}
