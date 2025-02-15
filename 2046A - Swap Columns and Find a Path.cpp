
/*#include<bits/stdc++.h>
using namespace std;


class DSU{
    public:
        vector<int>parent, size;
 
        DSU(int n){
            this->parent.resize(n);
            this->size.resize(n);
            for(int i=0; i<n; i++){
                parent[i]=i;
                size[i]=1;
            }
        }
 
        int findUParent(int node){
            if(parent[node]==node)return node;
            return parent[node] = findUParent(parent[node]);
        }
 
        void unionBySize(int a, int b){
            int Ulta = findUParent(a);
            int Ultb = findUParent(b);
            if(Ulta==Ultb)return;
            if(Ulta!=Ultb){
                if(size[Ulta]>=size[Ultb]){
                    parent[Ultb] = Ulta;
                    size[Ulta]+=size[Ultb];
                }
                else{
                    parent[Ulta] = Ultb;
                    size[Ultb]+=size[Ulta];
                }
            }
        }
};

int main()
{
    int t;
    cin >> t;
    while(t--)
    {
    int n,m1,m2 ;
    cin >> n ;
    cin >> m1;
    cin >> m2;

    DSU obj1(n);
    DSU obj2(n);
    vector<pair<int,int>> f(m1);
    vector<pair<int,int>> G(m2);
    for(int i=0;i<m1;i++)
    {
        int a,b ;
        cin >> a;
        cin  >>  b;
       f[i] = {a,b};
    }
    for(int j=0;j<m2;j++)
    {
        int a,b;
        cin >> a;
        cin >> b ;
        G[j] = {a,b};
        obj2.unionBySize(a-1,b-1);
    }
    int count =0;
    for(auto it : f)
    {
        if(obj2.findUParent(it.first-1)== obj2.findUParent(it.second-1))
        {
            obj1.unionBySize(it.first-1,it.second-1);
        }
        else
        {
            count++;
        }
    }
    int count1 =0;
    for(int i=0; i<n; i++){
        if(obj1.parent[i]==i)count1++;
    }
     int count2 =0;
   for(int i=0; i<n; i++){
        if(obj2.parent[i]==i)count2++;
    }
     count += (count1 - count2 );
    cout << count << endl;
}

}
*/
#include <iostream>
#include <bits/stdc++.h>
// #include <sys/resource.h>
// #include <ext/pb_ds/assoc_container.hpp>
// #include <ext/pb_ds/tree_policy.hpp>
 
using namespace std;
// using namespace chrono;
// using namespace __gnu_pbds;
 
// Speed
#define Code ios_base::sync_with_stdio(false);
#define By cin.tie(NULL);
#define Asquare cout.tie(NULL);
 
// def
// #define enableDebug 1
 
// Debug
#ifdef enableDebug
#define debug(x)       \
    cerr << #x << " "; \
    cerr << x << " ";  \
    cerr << endl;
#else
#define debug(x) ;
#endif
 
// Aliases
using ll = long long;
using lld = long double;
using ull = unsigned long long;
 
// Constants
const lld pi = 3.141592653589793238;
const ll INF = LONG_LONG_MAX;
const ll mod = 1e9 + 7;
const ll MOD = 1e9 + 7;
 
// TypeDEf
typedef pair<ll, ll> pll;
typedef vector<ll> vll;
typedef vector<pll> vpll;
typedef vector<string> vs;
typedef unordered_map<ll, ll> umll;
typedef map<ll, ll> mll;
 
// Macros
#define ff first
#define ss second
#define pb push_back
#define mp make_pair
#define fl(i, n) for (int i = 0; i < n; i++)
#define rl(i, m, n) for (int i = n; i >= m; i--)
#define py cout << "YES\n";
#define pm          \
    cout << "-1\n"; \
    s
#define pn cout << "NO\n";
#define vr(v) v.begin(), v.end()
#define rv(v) v.end(), v.begin()
 
// Operator overloads
template <typename T1, typename T2> // cin >> pair<T1, T2>
istream &operator>>(istream &istream, pair<T1, T2> &p)
{
    return (istream >> p.first >> p.second);
}
 
template <typename T> // cin >> vector<T>
istream &operator>>(istream &istream, vector<T> &v)
{
    for (auto &it : v)
        cin >> it;
    return istream;
}
 
template <typename T1, typename T2> // cout << pair<T1, T2>
ostream &operator<<(ostream &ostream, const pair<T1, T2> &p)
{
    return (ostream << p.first << " " << p.second);
}
 
template <typename T> // cout << vector<T>
ostream &operator<<(ostream &ostream, const vector<T> &c)
{
    for (auto &it : c)
        cout << it << " ";
    return ostream;
}
 
// Utility functions
template <typename T>
void print(T &&t) { cout << t << "\n"; }
void printarr(ll arr[], ll n)
{
    fl(i, n) cout << arr[i] << " ";
    cout << "\n";
}
 
template <typename T>
void printvec(vector<T> v)
{
    ll n = v.size();
    fl(i, n) cout << v[i] << " ";
    cout << "\n";
}
 
template <typename T>
ll sumvec(vector<T> v)
{
    ll n = v.size();
    ll s = 0;
    fl(i, n) s += v[i];
    return s;
}
 
// Mathematical functions
 
ll gcd(ll num1, ll num2)
{
    while (num2 != 0)
    {
        int rem = num1 % num2;
        num1 = num2;
        num2 = rem;
    }
 
    return num1;
} //__gcd
 
ll lcm(ll a, ll b) { return (a / gcd(a, b)) * b; }
 
ll moduloMultiplication(ll a, ll b, ll mod)
{
    ll res = 0;
    a %= mod;
    while (b)
    {
        if (b & 1)
            res = (res + a) % mod;
        b >>= 1;
    }
    return res;
}
 
ll powermod(ll x, ll y, ll p)
{
    ll res = 1;
    x = x % p;
    if (x == 0)
        return 0;
    while (y > 0)
    {
        if (y & 1)
            res = (res * x) % p;
        y = y >> 1;
        x = (x * x) % p;
    }
    return res;
}
 
// Sorting
bool sorta(const pair<int, int> &a, const pair<int, int> &b) { return (a.second < b.second); }
bool sortd(const pair<int, int> &a, const pair<int, int> &b) { return (a.second > b.second); }
 
// Bits
string decToBinary(int n)
{
    string s = "";
    int i = 0;
    while (n > 0)
    {
        s = to_string(n % 2) + s;
        n = n / 2;
        i++;
    }
    return s;
}
 
ll binaryToDecimal(string n)
{
    string num = n;
    ll dec_value = 0;
    int base = 1;
    int len = num.length();
    for (int i = len - 1; i >= 0; i--)
    {
        if (num[i] == '1')
            dec_value += base;
        base = base * 2;
    }
    return dec_value;
}
 
// Check
bool isPrime(ll n)
{
    if (n <= 1)
        return false;
    if (n <= 3)
        return true;
    if (n % 2 == 0 || n % 3 == 0)
        return false;
    for (int i = 5; i * i <= n; i = i + 6)
    {
        if (n % i == 0 || n % (i + 2) == 0)
            return false;
    }
    return true;
}
 
bool isPowerOfTwo(int n)
{
    if (n == 0)
        return false;
 
    return ((n & (n - 1)) == 0);
    // return (ceil(log2(n)) == floor(log2(n)));
}
 
bool isPerfectSquare(ll x)
{
    if (x >= 0)
    {
        ll sr = sqrt(x);
        return (sr * sr == x);
    }
    return false;
}
 
// Disjoint Set Union Template
class DSU
{
    vector<int> parent, size;
 
public:
    DSU(int n)
    {
        for (int i = 0; i <= n; i++)
        {
            parent.push_back(i);
            size.push_back(1);
        }
    }
 
    int findParent(int node)
    {
        if (parent[node] == node)
            return node;
 
        return (parent[node] = findParent(parent[node]));
    }
 
    void Union(int u, int v)
    {
        int pu = findParent(u);
        int pv = findParent(v);
 
        if (pu == pv)
            return;
 
        if (size[pu] < size[pv])
        {
            parent[pu] = pv;
            size[pv] += size[pu];
        }
 
        else
        {
            parent[pv] = pu;
            size[pu] += size[pv];
        }
    }
};
 
// Segment Tree Template
class SGTree
{
    vector<int> seg;
 
public:
    SGTree(int n)
    {
        int h = ceil(log2(n));
        int size = 2 * (int)pow(2, h);
        seg.resize(size + 1);
    }
 
    void build(int idx, int low, int high, vector<int> arr)
    {
        // leaf nodes
        if (low == high)
        {
            seg[idx] = arr[low];
            return;
        }
 
        int mid = (low + high) / 2;
        build(2 * idx + 1, low, mid, arr);
        build(2 * idx + 2, mid + 1, high, arr);
        seg[idx] = min(seg[2 * idx + 1], seg[2 * idx + 2]);
    }
 
    int query(int idx, int low, int high, int l, int r)
    {
        // no overlap
        // [l, r] [low, high] or [low, high] [l, r]
        if (r < low || high < l)
            return INT_MAX;
 
        // complete overlap
        // [l, low, high, r]
        if (l <= low && high <= r)
            return seg[idx];
 
        // partially overlap
        // [l, low, r, high] or [low, l, high, r]
        int mid = (low + high) / 2;
        int left = query(2 * idx + 1, low, mid, l, r);
        int right = query(2 * idx + 2, mid + 1, high, l, r);
        return min(left, right);
    }
 
    void update(int idx, int low, int high, int i, int val)
    {
        if (low == high)
        {
            seg[idx] = val;
            return;
        }
 
        int mid = (low + high) / 2;
        if (i <= mid)
            update(2 * idx + 1, low, mid, i, val);
 
        else
            update(2 * idx + 2, mid + 1, high, i, val);
 
        seg[idx] = min(seg[2 * idx + 1], seg[2 * idx + 2]);
    }
};
 
// Fenweek Tree Template
class FWTree
{
    int n;
    vector<int> fenweek;
 
public:
    FWTree(int size)
    {
        this->n = size;
        fenweek.resize(size + 1);
        fenweek[0] = 0;
    }
 
    void update(int idx, int val)
    {
        while (idx <= n)
        {
            fenweek[idx] += val;
            idx = idx + (idx & (-idx));
        }
    }
 
    int sum(int idx)
    {
        int s = 0;
        while (idx > 0)
        {
            s += fenweek[idx];
            idx = idx - (idx & -(idx));
        }
 
        return s;
    }
};
 
// SparseTable Template
class sparseTable
{
    int n;
    int size;
    vector<vector<int>> sparse;
 
public:
    sparseTable(int n)
    {
        this->n = n;
        this->size = log2(n) + 1;
 
        sparse.resize(n);
        for (int i = 0; i < n; i++)
            sparse[i] = vector<int>(size);
    }
 
    void build(vector<int> &arr)
    {
        for (int j = 0; j < size; j++)
        {
            for (int i = 0; (i + (1 << j) - 1) < n; i++)
            {
                if (j == 0)
                    sparse[i][j] = arr[i];
 
                else
                    sparse[i][j] = min(sparse[i][j - 1], sparse[i + (1 << (j - 1))][j - 1]);
            }
        }
    }
 
    int getMin(int l, int r)
    {
        int len = log2(r - l + 1);
        return min(sparse[l][len], sparse[r - (1 << len) + 1][len]);
    }
};
 
// Trie String Template
class TrieString
{
    struct Trienode
    {
        bool endsHere;
        Trienode *child[26];
    };
 
    Trienode *root;
 
    Trienode *getNode()
    {
        Trienode *newnode = new Trienode;
        newnode->endsHere = false;
 
        for (int i = 0; i < 26; i++)
            newnode->child[i] = NULL;
 
        return newnode;
    }
 
public:
    TrieString()
    {
        root = getNode();
    }
 
    void insert(string word)
    {
        Trienode *curr = root;
        int index;
 
        for (int i = 0; i < word.size(); i++)
        {
            index = word[i] - 'a';
            if (curr->child[index] == NULL)
                curr->child[index] = getNode();
 
            curr = curr->child[index];
        }
 
        curr->endsHere = true;
    }
 
    bool search(string word)
    {
        Trienode *curr = root;
        int index;
 
        for (int i = 0; i < word.size(); i++)
        {
            index = word[i] - 'a';
            if (curr->child[index] == NULL)
                return false;
 
            curr = curr->child[index];
        }
 
        return curr->endsHere;
    }
};
 
// Trie Xor Template
class TrieXor
{
    struct Trienode
    {
        Trienode *child[2];
    };
 
    Trienode *root;
 
    Trienode *getNode()
    {
        Trienode *newnode = new Trienode;
        newnode->child[0] = NULL;
        newnode->child[1] = NULL;
 
        return newnode;
    }
 
public:
    TrieXor()
    {
        root = getNode();
    }
 
    void insert(int num)
    {
        Trienode *curr = root;
        int bit;
        for (int i = 31; i >= 0; i--)
        {
            bit = (num >> i) & 1;
            if (curr->child[bit] == NULL)
                curr->child[bit] = getNode();
 
            curr = curr->child[bit];
        }
    }
 
    int getMax(int num)
    {
        int maxNum = 0;
        Trienode *curr = root;
        int bit;
 
        for (int i = 31; i >= 0; i--)
        {
            bit = (num >> i) & 1;
 
            if (curr->child[1 - bit] != NULL)
            {
                maxNum |= (1 << i);
                curr = curr->child[1 - bit];
            }
 
            else
            {
                curr = curr->child[bit];
            }
        }
 
        return maxNum;
    }
};
 
// Hash Function
long long computeHash(string &s)
{
    // Here we take parime as 31
    long long p = 1;
    long long hash = 0;
 
    for (char ch : s)
    {
        hash = (hash + (((ch - 'a' + 1) * p) % MOD)) % MOD;
        p = (p * 31) % MOD;
    }
 
    return hash;
}
 
ll power(ll base, ll n, ll MOD)
{
    ll ans = 1;
    while (n)
    {
        if (n & 1)
        {
            ans = ((ans * base) % MOD);
            n = n - 1;
        }
 
        else
        {
            n = n >> 1;
            base = ((base * base) % MOD);
        }
    }
 
    return ans;
}
 
// Prefix Hashing
class prefixHashing
{
 
    vector<long long> prefixHash;
 
public:
    prefixHashing(string &s)
    {
        int n = s.size();
        long long p = 1;
        long long hash = 0;
 
        for (int i = 0; i < n; i++)
        {
            hash = (hash + (((s[i] - 'a' + 1) * p) % MOD)) % MOD;
            prefixHash.push_back(hash);
            p = (p * 31) % MOD;
        }
    }
 
    long long computeSubstringHash(int l, int r)
    {
        long long hash = prefixHash[r];
        if (l - 1 >= 0)
            hash = (hash - prefixHash[l - 1] + MOD) % MOD;
 
        hash = (hash * power(power(31, l, MOD), MOD - 2, MOD)) % MOD;
        return hash;
    }
};
 
// Rabin Karp for string matching algo
int rabinKarp(string &s, string &pattern)
{
    int n1 = s.size();
    int n2 = pattern.size();
    int ans = 0;
 
    // compute patten hash
    long long patHash = 0;
    long long p = 1;
    for (char ch : pattern)
    {
        patHash = (patHash + (((ch - 'a' + 1) * p) % MOD)) % MOD;
        p = (p * 31) % MOD;
    }
 
    // compute hash for first window of s
    long long hash = 0;
    long long p1 = 1;
    long long p2 = 1;
    for (int i = 0; i < n2; i++)
    {
        hash = (hash + (((s[i] - 'a' + 1) * p2) % MOD)) % MOD;
        p2 = (p2 * 31) % MOD;
    }
 
    int left = 0, right = n2;
    if (hash == patHash)
        ans++;
 
    while (right < n1)
    {
        hash = (hash + (((s[right] - 'a' + 1) * p2) % MOD)) % MOD;
        hash = (hash - (((s[left] - 'a' + 1) * p1) % MOD) + MOD) % MOD;
        p2 = (p2 * 31) % MOD;
        p1 = (p1 * 31) % MOD;
        right++;
        left++;
 
        patHash = (patHash * 31) % MOD;
 
        if (patHash == hash)
            ans++;
    }
 
    return ans;
}
 
// Longest Prefix as Suffix Construction
vector<int> LPS(string &s)
{
    int n = s.size();
    int i = 1, len = 0;
 
    vector<int> lps(n, 0);
 
    while (i < n)
    {
        if (s[i] == s[len])
        {
            len++;
            lps[i] = len;
            i++;
        }
 
        else
        {
            if (len != 0)
                len = lps[len - 1];
 
            else
            {
                lps[i] = 0;
                i++;
            }
        }
    }
 
    return lps;
}
 
// KMP String Pattern matching algo
vector<int> KMP(string &s, string &pattern)
{
    int n1 = s.size();
    int n2 = pattern.size();
 
    vector<int> lps = LPS(pattern);
    vector<int> res;
    int i = 0, j = 0;
    while (i < n1)
    {
        if (s[i] == pattern[j])
        {
            i++;
            j++;
        }
 
        else
        {
            if (j == 0)
                i++;
            else
                j = lps[j - 1];
        }
 
        if (j == n2)
        {
            res.push_back(i - n2);
            j = lps[j - 1];
        }
    }
 
    return res;
}
 
// Z algo(function)
vector<int> Z_function(string &s)
{
    int n = s.size(), l = 0, r = 0;
    vector<int> z(n, 0);
    for (int i = 1; i < n; i++)
    {
        if (i <= r)
            z[i] = min(z[i - l], r - i + 1);
 
        while (i + z[i] < n && s[z[i]] == s[z[i] + i])
            z[i]++;
 
        if (i + z[i] - 1 > r)
        {
            l = i;
            r = i + z[i] - 1;
        }
    }
 
    return z;
}
 
void solve()
{
    ll n;
    cin >> n;
 
    vector<ll> arr(n + 1);
    for (ll i = 1; i <= n; i++)
        cin >> arr[i];
 
    ll ans = 0;
    for (ll i = 1; i <= n; i++)
    {
        for (ll j = (arr[i] - i); j <= n; j += arr[i])
        {
            if (j >= 0 && i < j && arr[i] * arr[j] == i + j)
                ans++;
        }
    }
 
    cout << ans << "\n";
}
void solve1()
{
    ll n,k,s;
    cin>>n>>k>>s;
    ll arr[n];
    for(ll i=0;i<n;i++)
        cin>>arr[i];
    sort(arr,arr+n);
    vector<ll> v;
    for(ll i=1;i<n;i++){
        if(arr[i]-arr[i-1]>s){
            v.push_back((arr[i]-arr[i-1]-1)/s);
        }
    }
    sort(v.begin(),v.end());
   int count =0;
     for(int i=0;i<v.size();i++)
     {
        if(v[i]<=k){
            k-=v[i];
            
        }
        else
            
            {
                count = v.size()-i;
                break;
            }
    }
    cout<<count+1<<endl;   


}
void solve2()
{
    int n ;
    cin >> n ;
    vector<int> v(n);
     unordered_map<int,int> mp;
     long long count =0;
    for(int i=0;i<n;i++)
    {
        cin >> v[i];
        if(mp.find(v[i] -i) == mp.end())
        {
        mp[v[i] - i]++;
        }
        else{
            count = count+ mp[v[i] - i];
            mp[v[i] - i]++;
            
             
        }
    }
    cout << count << endl;
 
  
}
void solve3()
{
    ll n,k;
    cin >> n;
    cin >> k;
    vector<ll> v(n);
    ll sum =0;
    ll maxi =0;
    for(int i=0;i<n;i++)
    {
        cin >> v[i];
        sum += v[i];
        maxi = max(maxi,v[i]);
    }
 ll ans =0;
    for(int i= n;i>=0;i--)
    {
        ll m = (sum+k)/i ;
       
        if( m >= maxi &&  sum <= m*i)
        {
          ans = i;
          break;  
        }

    }
    cout << ans << endl ;

}
long long dfs1(int node,int par,long long d,vector<vector<int>> &adj,vector<long long> &sol)
{
    long long sub_size =0;
    for(auto it : adj[node])
    {
        if(it == par)continue ;
        sub_size += dfs1(it,node,d+1,adj,sol);
    }
    sol[node] = d - sub_size;
    return sub_size+1;
}
void solve4()
{
    // Linova and Kingdom // very very imp 
    int n,k;
    cin >> n;
    cin >> k ;
    vector<vector<int>> adj(n+1);
    for(int i=0;i<n-1;i++)
    {
      int a,b;
      cin >> a;
      cin >> b;
      adj[a].push_back(b);
      adj[b].push_back(a);
    }

    vector<long long> sol(n+1);
    dfs1(1,0,0,adj,sol);
    sort(sol.begin()+1,sol.begin()+n+1);
    long long ans =0;
    for(int i=0;i<k;i++ )
    {
        ans += sol[n-i];
    }
    cout << ans << endl;
}

void solve5()
{
    int n,k,z;
    cin >> n;
    cin >> k;
    cin >> z;
    vector<int> v(n);
    for(int i=0;i<n;i++)
    {
        cin >> v[i];
    }
    int ans =0;
    for(int i=0;i<=z;i++)
    {
        int rem = k- 2*i;
        if(rem <0)
        continue ;
        int maxi =0;
        int temp =0;
        for(int j=0;j<=rem;j++)
        {
            if(j<n-1)
            {
           maxi = max(maxi,v[j]+v[j+1]);
            }
             temp += v[j];
        }
        ans = max(ans,temp+(maxi*i));
    }
    cout << ans << endl;
}
  void solve6()
  {
    int n ;
    cin >> n ;
    vector<long long> v(n);
    for(int i=0;i<n;i++)
    {
        cin >> v[i];
    }
    sort(v.begin(),v.end());
    int i=0;
    int j = n-1 ;
    long long x =0;
    long long count =0;
    while(i<=j)
    {
        if(v[j] > x)
        {
            int t = min(v[i],v[j]-x);
            if(i == j)
            {
                if(v[i] == 1)
                
                {
                    count++;
                    
                }
        else {
            count += (v[j] - x - 1) / 2 + 1 + 1;
        }
                
                break;
            
            }
                count = count + t;
                x = x+ t;
                v[i]=v[i]-t;
                if(v[i] == 0)
                i++;

        }
        else{
            count++;
            j--;
            x =0;

        }
    }
    cout << count << endl;
  }
  void solve7()
  {
    int n ;
    cin >> n;
    vector<long long>v(n);
    for(int i=0;i<n;i++)
    {
        cin >> v[i];
    }
    long long s =0;
    unordered_map<long long,long long> mp;
    mp[0]=1;
    for(int i=0;i<n;i++)
    {
        if(i%2!=0)
        {
            v[i]=-v[i];
        }
        s = s +v[i];
        if(mp.find(s)!=mp.end())
        {
            cout << "YES" << endl;
            return;
        }
        mp[s]++;

    }
    cout << "NO" << endl;
  }
  void solve8()
  {
    int n ;
    cin >> n ;
    vector<long long> a(n);
    vector<long long > b(n);
    for(int i=0;i<n;i++)
    {
        cin >> a[i];
    }
    for(int i=0;i<n;i++)
    {
        cin >> b[i];
    }
    vector<long long> prefix(n+1);
   
    for(int i=0;i<n;i++)
    {
        prefix[i+1] = prefix[i] + b[i];
    }

    vector<long long> left(n+1,0);
    vector<long long> count(n+1,0);
    for(int i=0;i<n;i++)
    {
        int x = upper_bound(prefix.begin(),prefix.end(),prefix[i] + a[i])-prefix.begin()-1;
        count[i] = count[i]+1;
        count[x] = count[x]-1;
        left[x] += a[i]-prefix[x] + prefix[i];


    }
  
    for(int i=0;i<n;i++)
    {
       cout << count[i]*b[i] + left[i] << " ";
       count[i+1] += count[i];
    }
    cout << endl;
  }
  void solve9()
  {

  int n ;
  cin >> n ;
 
  vector<int> b(n);
  unordered_map<int,int> mp;
  for(int i=0;i<n;i++)
  {
    int x;
    cin >> x;
    mp[x] = i;
  }
  for(int i=0;i<n;i++)
  {
    cin>> b[i];

  } 
  int count =1;
  for(int i=n-1;i>0;i--)
  {
    if(mp[b[i]] > mp[b[i-1]])
    {
        count++;
    }
    else{
        break;
    }
  }
  cout << n - count << endl;
  }
  void solve10()
  {
    
		int n;
		cin >> n;
		vector<int> a(n);
		for (auto &it : a) cin >> it;
		vector<pair<int, int>> res;
		int idx = -1;
		for (int i = 1; i < n; ++i) {
			if (a[i] != a[0]) {
				idx = i;
				res.push_back({1, i + 1});
			}
		}
		if (idx == -1) {
			cout << "NO" << endl;
			return;
		}
		for (int i = 1; i < n; ++i) {
			if (a[i] == a[0]) {
				res.push_back({idx + 1, i + 1});
			}
		}
		cout << "YES" << endl;
		for (auto [x, y] : res) cout << x << " " << y << endl;
	}
	
  void solve11()
  {
    ll n ;
    cin >> n ;
    vector<pair<ll,ll>> v(n);
    for(int i=0;i<n;i++)
    {
        cin >> v[i].second >> v[i].first ;
    }
    sort(v.begin(),v.end());
    int i=0;
    int j = n-1;
    ll count =0;
    ll cost =0;
    while(i<=j)
    {
       if(count < v[i].first)
       {
        ll x = v[j].second;
        if(v[i].first-count < v[j].second)
        {
           x = v[i] .first - count ;
           
        }
      //  int x = min(v[j].second,v[i].first - count);
        count = count+ x;
        cost = cost + x*2;

             
       

        v[j].second = v[j].second - x;
        if(v[j].second == 0)
        j--;


       }
       else{
        count = count + v[i].second;
        cost = cost + v[i].second ;
        i++;
       }
    }
    cout << cost << endl;
  }
  
  void solve12()
  {
    int n ;
    cin >> n ;
    vector<int> v(n);
    int  mini = INT_MAX ;
    int ind = -1 ;
    for(int i=0;i<n;i++)
    {
        cin >> v[i];
        if(v[i]<mini)
        {
            ind = i;
            mini = v[i];
        }
    }
    for(int i=ind+1;i<n;i++)
    {
        if(v[i] < v[i-1])
        {
            cout << -1 << endl;
            return;
        }
    }
    cout << ind << endl;

  }
  void solve13()
  {
    int n,q;
    cin >> n;
    cin >> q ;
  unordered_map<int,int> dist;
  int t=0;
    for(int  i=0;i<n-1;i++)
    {
        cout << i+1 << " " << i+2 << endl;
        dist[i+1] = i+1;
        dist[i+2] = i+2;
       
    }

  int cur = n-1 ;
  int ind = n-1;
    for(int i=0;i<q;i++)
    {
        int x;
        cin >> x;
        if(x == cur)
        {
            cout << -1 << " " << -1 << " " << -1 << endl;
        }
        else{
               cout << n <<  " " << ind << " " << dist[x] << endl;
               ind = dist[x];
              
               cur = x;
        }

    }
  }
  void solve14()
  {
    ll n ;
    cin >> n ;
    vector<ll> v(n);
    for(int i=0;i<n;i++)
    {
        cin>> v[i];
    }
    sort(v.begin(),v.end());
    long long ans = 0;
    long long sum =0;
    for(int i=0;i<n;i++)
    {
      sum += v[i];
      ans += sum - (long long)(1+i)*v[i];
    }
    ans += v[n-1];
    cout << ans << endl;
  }
  void solve15()
  {
    int n ;
    cin >> n;
    vector<long long> v(n);
    for(int i=0;i<n;i++)
    {
        cin >> v[i];
    }
    int count_pos =0;
    long long even =0;
    long long odd =0;
    long long maxi = INT_MIN;
    for(int i=0;i<n;i++)
    {
          if(v[i] > 0)
        count_pos++;
        maxi = max(maxi,v[i]);

        if(i%2==0 && v[i] > 0)
        even += v[i];
        else if(v[i] > 0 && i%2!=0)
        odd += v[i];

        
    }

    if(count_pos ==0)
    {
        cout << maxi << endl;
        return;
    }

    cout << max(even,odd) << endl;

  }
  void solve16()
  {
    ll n;
    cin >> n;
    vector<set<ll>> graph(n + 1, set<ll>());
    vector<ll> degree(n + 1, 0);
    
    for (ll i = 1; i < n; i++) {
        ll x, y;
        cin >> x >> y;
        graph[x].insert(y);
        graph[y].insert(x); 
        degree[x]++;
        degree[y]++;
    }
    vector<pair<ll,ll>>choice;
    for(ll i=1;i<=n;i++){
        choice.push_back({degree[i],i});
    }
    sort(choice.begin(),choice.end(),greater<pair<ll,ll>>());
    
    
    ll ans=0;
    for(ll node=1;node<=n;node++){
        for(auto nextnode:graph[node]){
            ll score=degree[node]+degree[nextnode]-2;
            ans=max(ans,score);
        }
    }
   
   
    for(ll node=1;node<=n;node++){
        for(ll i=0;i<choice.size();i++){
            ll nxtNode=choice[i].second;
            if(graph[node].find(nxtNode)==graph[node].end()&&node!=nxtNode){
                ll score=degree[node]+degree[nxtNode]-1;
                ans=max(ans,score);
                break;
            }
            else{
                continue;
            }
        }
    }
   
    cout<<ans<<endl;
    

  }
  void solve17()
  {
    int y,k,n,x;
	cin>>y>>k>>n;
	x=k-y%k;
	if(x+y>n)cout<<-1;
	else 
    {
    while(x+y<=n)
    {  
        cout<<x<<' ';
        x = x+k;
    }
    }
  }
  void solve18()
  {
    int n,w;
    cin >> n ;
    cin >> w;
    vector<int> v(n);
    for(int i=0;i<n;i++)
    {
        cin >> v[i];
    }
    unordered_map<int,int> mp;
    int maxi =0;
    for(int i=0;i<n;i++)
    {
        mp[v[i]]++;
        maxi = max(maxi,mp[v[i]]);
    }
    cout << maxi << endl;
  }

  void solve19()
  {
    
     ll n,w;
     cin>>n>>w;
     vll v(n);
     for (auto &e:v)cin>>e;     
      ll ans=0;
      sort(v.begin(),v.end());
     while(v.size()){
        int a=w;
        while(a){
        int i=upper_bound(v.begin(),v.end(),a)-v.begin()-1;
        if(i>=0&&i<v.size()){a-=v[i];v.erase(v.begin()+i);}
        else break;
        }
        ans++;
     }
     cout<<ans<<endl;
     

  }
  void solve20()
  {
    int n ;
    cin >> n ;
    
    vector<pair<int,int>> b;
    for(int i=0;i<n;i++)
    {
   int k;
    cin >> k ;
    int t=0;
    int maxi =0;
    for(int j=0;j<k;j++)
    {
        int x;
        cin >> x;
        maxi = max(maxi,x-t);
        t++;
    }
  b.push_back({maxi,k});
    }

    sort(b.begin(),b.end());
    int count =0;
    int ans =0;
    for(int i=0;i<n;i++)
    {
     ans = max(ans,b[i].first-count);
     count = count +  b[i].second;
    }
    cout << ans+1<< endl;
  }
  void solve21()
  {
    int n ;
    cin >> n ;
    vector<int> v(n);
    for(int i=0;i<n;i++)
    {
        cin >> v[i];
    }
    vector<int> ans ;
    ans.push_back(v[0]);
     
    for(int i=1;i<n-1;i++)
    {
        if((v[i-1] < v[i]) != (v[i] < v[i+1]))
        {
            ans.push_back(v[i]);
        }
    }
    ans.push_back(v[n-1]);

cout << ans.size() << endl;
for(auto it : ans)
{
    cout << it << " ";
}
cout << endl;
  }
  void solve22()
  {
    int n ;
    cin >> n ;
    vector<int>a(n);
    vector<int> b(n);
    unordered_map<int,int> mp;
    for(int i=0;i<n;i++)

    {
        cin >> a[i];
    }
    for(int j=0;j<n;j++)
    {
          cin >> b[j];
      mp[b[j]] = j;
    }
    int maxi =mp[a[0]];
    int defaulter =0;
    for(int i=1;i<n;i++)
    {
      if(maxi > mp[a[i]])
      {
        defaulter++;
      }
      maxi = max(maxi,mp[a[i]]);
    }
    cout << defaulter << endl;


  }
  void solve23()
  {
    int n ;
    cin >> n ;
    vector<int> v(n);
   int start =0;
   int end =0;
   
    for(int i=0;i<n;i++)
    {
        cin >> v[i];

         
    }
    start =0;
    int store1 = v[0];
    int wide =0;
    for(int i=1;i<n;i++)
    {
       if(v[i]!= store1)  
       wide = i;
    }
    int back_wide =0;
    end = n-1;
    int store2 =v[n-1];
    for(int i=n-2;i>=0;i--)
    {
        if(v[i]!=store2)
        {
            back_wide = i;
        }
    }
    int ans = max(wide,n-1-back_wide);
    cout << ans << endl;
  }
  bool cal(int &n,vector<pair<int,int>> &v,vector<int> &queries,int &mid)
  {
    vector<int> pre_0(n+1,0);
    vector<int> pre_1(n+1,0);
    vector<int> arr(n+1,0);
    for(int i=1;i<=mid;i++)
    {
        arr[queries[i]] = 1;
    }
    for(int i=1;i<=n;i++)
    {
      pre_0[i] = pre_0[i-1] + !arr[i];
      pre_1[i] = pre_1[i-1] + arr[i] ;
    }
     int m = v.size();

    for(int i =0;i<m;i++)
    {
          if(pre_1[v[i].second] - pre_1[v[i].first -1] > pre_0[v[i].second] - pre_0[v[i].first-1])
          {
            return true ;
          }
    }
    return false ;
  }
  void solve24()
  {
    int n,m;
    cin >> n;
    cin >> m ;
    vector<pair<int,int>> v(m);
    for(int i=0;i<m;i++)
    {
        cin >> v[i].first ;
        cin >> v[i].second ;
    }
    int q;
    cin >> q;
    vector<int> queries(q+1);
    for(int i=1;i<=q;i++)
    {
        cin >> queries[i];
    }
    int i=1;
    int j = q;
    bool flag =false ;
    int mini = INT_MAX ;
    while(i<=j)
    {
        int mid = i+ (j-i)/2;
        if(cal(n,v,queries,mid))
        {
          flag = true ;
          mini = min(mini,mid);
           j = mid-1;
        }
        else{
            i = mid+1;
        }
    }
    if(flag)
    {
        cout << mini << endl;
    }
    else{
        cout << -1 << endl;
    }

  }
  void solve25()
  {
    int n;cin>>n;
    vector<int> v(n);
    
    set<int>st;
    for (int i = 0; i < n; ++i)
    {
        cin>>v[i];
        st.insert(v[i]);
    }
    int mn=*min_element(v.begin(), v.end());
    int ev=0,od=0;
   
    int on=0;
    for(auto it:st)
    {
        if(it==1)on++;
        if(it%2==0)ev++;
        else od++;
    }
   
 
    if(mn%2!=0||ev==0||od==0)cout<<"YES"<<endl;else cout<<"NO"<<endl;
  }
 
 void solve26()
 {
    int n,k;
    cin >> n;
    cin >> k;
    vector<pair<int,int>> v1(n);
    vector<int> v2(n);
    for(int i=0;i<n;i++)
    {
        int x;
        cin >> x;
        v1[i] = {x,i};
    }
    for(int i=0;i<n;i++)
    {
        cin >> v2[i];
    }
    sort(v1.begin(),v1.end());
    sort(v2.begin(),v2.end());
    vector<int> ans(n);
    for(int i=0;i<n;i++)
    {
        ans[v1[i].second] = v2[i];
    }
    for(int i=0;i<n;i++)
    {
        cout << ans[i] << " ";
    }
    cout << endl;
 }
 void solve27()
 {
    ll n ,v1,v2;
    cin >> n;
    cin >> v1;
    cin >> v2;
    v1--;
    v2--;
    vector<vector<ll>> G1(n);
    vector<vector<ll>> G2(n);
 ll m1;
    cin >> m1;
    for(int i=0;i<m1;i++)
    {
        int a,b;
        cin >> a;
        cin >> b;
   a--;
   b--;
        G1[a].push_back(b);
        G1[b].push_back(a);
    }
    ll m2;
    cin >> m2;
    for(int i=0;i<m2;i++)
    {
        ll a,b;
        cin >> a;
        cin >> b;
        a--;
        b--;
        G2[a].push_back(b);
        G2[b].push_back(a);

    }

    vector<bool> req_edge(n,false);
    // let this node be common now check for some common edge
    for(int i=0;i<n;i++)
    {
      set<ll> neigh;
      for(auto it : G1[i])
      {
        neigh.insert(it);
      }
      for(auto it : G2[i])
      {
        if(neigh.find(it)!= neigh.end())
        {
              req_edge[i] = true ;
              break;
        }
      }
    }

     priority_queue<vector<ll>, vector<vector<ll>>, greater<vector<ll>>> pq;
    pq.push({0,v1,v2});
    vector<vector<ll>> dist(n,vector<ll>(n,1e18));
    dist[v1][v2] = 0;
    int ans = -1;
    while(!pq.empty())
    {
        auto it = pq.top();
        pq.pop();
        int d = it[0];
        int x = it[1];
        int y = it[2] ;

         if(d != dist[x][y])
         continue ;
         if(x==y && req_edge[x])
         {
          ans = d;
          break;
         }

         for(auto it1 : G1[x])
         {
            for(auto it2 : G2[y])
            {
                ll diff =  d + abs(it1 - it2);
                if(dist[it1][it2] > diff)
                {
                    dist[it1][it2] = diff;
                    pq.push({diff,it1,it2});
                }
            }
         }
    }
    cout << ans << endl;

 }
 void solve28()
 {
    int n;
    cin >> n;
    vector<int> v(n);
    for(int i=0;i<n;i++)
    {
        cin >> v[i];
    }
    vector<set<int>> adj(n);
    for(int i=0;i<n;i++)
    {
        adj[i].insert(v[i]-1);
        adj[v[i]-1].insert(i);

    }
    vector<int> edges(n);
    for(int i=0;i<n;i++)
    {
        edges[i] = adj[i].size();

    }
    int cycles =0;
    int bamboo =0;
    vector<int> vis(n,0);
    for(int i=0;i<n;i++)
    {
        if(vis[i] == 0)
        {
            queue<int> q;
            q.push(i);

        set<int> comp;
        comp.insert(i);
        vis[i] = 1;
        while(!q.empty())
        {
            int node = q.front();
            q.pop();
            for(auto it : adj[node])
            {
                if(vis[it] == 0)
                {
                    vis[it] = 1;
                    q.push(it);
                    comp.insert(it);
                }
            }
        }
        bool b = false ;
        for(auto it : comp)
        {
            if(edges[it] ==1)
            {
                b = true ;
                break; 
            }
        }
        if(b)
        {
            bamboo++;
        }
        else{
            cycles++;
        }

        }
        
    }
    cout << (cycles + min(1,bamboo)) << " " << (cycles + bamboo) << endl;
 }

 bool perform(vector<ll>& a, ll x, ll mid) {
    ll total = x;
    for (ll height : a) {
        if (mid < height) {
            continue ;
        }
        total -= (mid - height);
    }
    return total >= 0;
}

 void solve29()
 {
    int n ;
    cin >> n ;
  int x;
  cin >> x;
       vector<ll> v(n);
    for (auto &u : v) {
        cin >> u;
    }
    ll left = 1;
    ll right = 1e10;
    ll ans = 0;
    while (left <= right) {
        ll mid = (left + right) >> 1;
        if (perform(v, x, mid)) {
            ans  = mid;
            left = mid + 1;
        } else {
            right = mid-1;
        }
    }
    cout << ans << endl;
 }
 void solve30()
 {
     int n, x;
        cin >> n >> x;
        int s = 0;
        vector<int> a(n);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < n; j++) cin >> a[j];
            for (int j = 0; j < n; j++) {
                if ((x | a[j]) != x) break;
                s |= a[j];
            }
        }
        if (s == x) cout << "YES\n";
        else cout << "NO\n";
 }
 void solve31()
 {
    int n ,k;
    cin>>n>>k;
    vector<long long> a(n+1);
    a[0] = 0;
for(int i=1;i<=n;i++) cin>>a[i];
sort(a.begin(),a.end());
for(int i=2;i<=n;i++)
a[i]+=a[i-1];
long long ans=0;
for(int i=0;i<=k;i++)
ans=max(ans,a[n-k+i]-a[i*2]);
cout<<ans<<endl;
 }
 void solve32()
 {

		int n;
		cin>>n;
		vector<int> v(n);
		for(int i=0;i<n;i++) cin>>v[i];
		int m=-1,ans=1,u;
		for(int i=1;i<n;i++){
			if(v[i]>v[i-1]) u=1;
			else if(v[i]<v[i-1]) u=0;
			else continue;
			if(u!=m) ans++;
			m=u;
        }
		cout<<ans<<endl;
    
 }
 void solve33()
 {
    int n,m;
    cin >> n;
    cin >> m ;
    vector<int> v(n);
    int even =0;
    int odd =0;
    priority_queue<int,vector<int>,greater<int>> pq;
    for(int i=0;i<n;i++)
     cin >> v[i];
    for(int i=0;i<n-1;i++)
    {
       
        if(v[i]%2 ==0)
        {
            even++;
        }
        else
        odd++ ;


        if(even==odd)
        pq.push(abs(v[i+1] - v[i]));

    }
    int count =0;
     while(!pq.empty())
     {

        auto it = pq.top();
        pq.pop();
        m = m-it;
        if(m>=0)
        count++;
        else
        break;

     }
     cout << count << endl;
 }
 void solve34()
 {
    
    ll n,k;
     cin>>n>>k;
     
     vector<ll> a(n),b(n),p(n);
     ll ans=0;   
     for(int i=0;i<n;i++){cin>>a[i];ans+=a[i];}
     for(int i=0;i<n;i++){
        cin>>b[i];
        p[i]=a[i] - b[i];
     }
    sort(p.begin(),p.end());
     for(ll i=n-1;i>=k;i--){
        ans-=(max((ll)0,p[i]));
     }
     cout<<ans<<endl;
 }
 
 void solve35()
 {
    ll k;
     string s;
     cin>>s>>k;
     ll ans=0;
     ll n=s.size();
     for(int i=n-1;i>=0;i--)
     {
         if(k<=0)
         {
             break;
         }
         if(s[i]=='0')
         {
             k--;
         }
         else
         {
             ans++;
         }
     }
     if(k<=0)
     {
         cout<<ans<<endl;
     }
     else
     {
         cout<<s.size()-1;
     }
 
 }

 void solve36()
 {
    int n ;
    cin >> n ;
    vector<vector<int>> v(n,vector<int>(n));
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            cin >> v[i][j];
        }
    }
    vector<int> suffix;
    for(int i=0;i<n;i++)
    {
        int count =0;
        for(int j=n-1;j>=0;j--)
        {

            if(v[i][j] == 1)
            {
                count++;
            }
            else{
                break;
            }
        }
        suffix.push_back(count);
    }
    sort(suffix.begin(),suffix.end());
    int maxi =0;
    for(auto it : suffix)
    {
        if(it>=maxi)
        {
            maxi++;
        }
    }
    cout << maxi << endl;
 }
 /* const int N=1e6;
int t,n,a[N],p[N],s[N],A,B,C,l;
int solve(int l,int r)
{
	if((p[r]-p[l-1])%2==0&&s[r]-s[l-1]>C)
		C=s[r]-s[l-1],A=l-1,B=n-r;
}
 void solve37()
 {
    
		for(int i=1;i<=n;i++)
			cin>>a[i],p[i]=p[i-1]+(a[i]<0),s[i]=s[i-1]+(abs(a[i])>1);
		A=n,B=C=l=a[n+1]=0;
		for(int i=1;i<=n+1;i++)
			if(a[i]==0)
			{
				for(int j=l+1;j<i;j++)
					solve(l+1,j),solve(j,i-1);
				l=i;
			}
		cout<<A<<' '<<B<< endl;
	

 }
 */
 bool disambuish(int &mid,vector<pair<int,int>> &v,int &n)
 {
    int no_of_peop = 0;
       for(int i=0;i<n;i++)
       {
        if(no_of_peop <= v[i].second &&  ((mid-1-no_of_peop) <= v[i] .first))
        {
            no_of_peop++;
        }
       }
       if(no_of_peop >= mid)
       return true;
       return false ;
 }
 void solve38()
 {
    int n ;
    cin >> n ;
    vector<pair<int,int>> v(n);
    for(int i=0;i<n;i++)
    {
        cin >> v[i].first;
        cin >> v[i].second;
    }
    int i=1;
    int j=n;
    int ans =-1;
    while(i<=j)
    {
        int mid = i+(j-i)/2;
        if(disambuish(mid,v,n))
        {
       ans = mid;
       i = mid+1;
        }
        else{
            j = mid-1;
        }
    }
    cout << ans << endl;

 }
 void solve39()
 {
   
    ll n; cin>>n;
    double a[n],b[n],maxi=INT_MIN,mini=INT_MAX;
    for(ll i=0;i<n;i++) cin>>a[i];
    for(ll i=0;i<n;i++) cin>>b[i];
    for(ll i=0;i<n;i++)
    {
      maxi=max(maxi,a[i]+b[i]);
      mini=min(mini,a[i]-b[i]);
    }
    cout<<fixed<<setprecision(1)<<(maxi+mini)/2<<endl;
 
 }
 void solve40()
 {
    ll a,b,n;
    cin >> a;
    cin >> b;
    cin >> n;
    vector<ll> v(n);
    ll ans =b;
    for(int i=0;i<n;i++)
    {
        cin >> v[i];
        ans = ans + min(v[i],a-1);
    }
    
cout << ans << endl;
 }
 void solve41()
 {
    int n,k;
    cin >>n ;
    cin >> k ;
   vector<int> v(n);
   for(int i=0;i<n;i++)
   {
    cin >> v[i];
   }
   sort(v.begin(),v.end());
   int ans = 1;
    int count =1;
   for(int i=1;i<n;i++)
   {
    if(v[i] - v[i-1]  <= k)
    {
        count++;

    }
    else{
         count = 1;
    }
    ans = max(ans,count);
   }
   cout << n - ans << endl;
 }
  void dfs(int node,int par,int col,vector<long long> &cnt,vector<vector<long long>> &adj)
  {
    cnt[col]++;
    for(auto it : adj[node])
    {
        if(it!=par)
        {
            dfs(it,node,!col,cnt,adj);
        }
    }
  }
 void solve42()
 {
    long long n;
    cin >> n ;
    vector<vector<long long>> adj(n+1);
    for(int  i=0;i<n-1;i++)
    {
        long long a,b;
        cin >> a;
        cin >> b;
  adj[a].push_back(b);
  adj[b].push_back(a);
    }
    vector<long long> cnt(2);
    dfs(1,0,0,cnt,adj);
    long long ans = cnt[0] *cnt[1] ;
     cout << ans-n+1 << endl;
 }
 void solve43()
 {
    int n,q;
    cin >> n;
    cin >> q;
    vector<int> a(n);
    vector<int> b(q);
    for(int i=0;i<n;i++)
    {
        cin >> a[i];

    }

    for(int i=0;i<q;i++)
    {
        cin >> b[i];
        int ans = min(b[i],a[0]-1);
        cout << ans << " ";
    }
    cout << endl;
 }
 void solve44()
 { int n,k;
    cin>>n>>k;
    vector<int>a(n+1);
    vector<int> s(n+1);
		for(int i=1;i<=n;i++){
			cin>>a[i];
			s[i]=s[i-1]+a[i];
		}
        vector<int> h(n+1);
		for(int i=1;i<=n;i++)cin>>h[i];
		int maxi=0;
		int l=1;
        int r=1;
		while(r<=n){
			if(h[r-1]%h[r])l=r;
			while(s[r]-s[l-1]>k)l++;
			maxi=max(maxi,r-l+1);
			r++;
		}
		cout<<maxi<<endl;
 }
 void solve45()
 {
    int x,y;
    cin >> x;
    cin >> y;
    if( y == x+1 || ( x>y && (x-y+1)%9 == 0))
    {
        cout << "YES" <<  endl;
    }
    else
    cout << "NO" << endl;
 }
 void solve46()
 {
    int n;
    cin >> n ;
    vector<int> v(n);
    for(int i=0;i<n;i++)
    {
        cin >> v[i];
    }
    unordered_map<int,int> mp;
    for(int i=0;i<n;i++)
    {
        mp[v[i]]++;
    }
    for(int i= 1;i<=n;i++)
    {
        if(mp[i] >2)
        {
            mp[i+1] += mp[i] -2;
          mp[i] = 2;
        }

    }
    bool b = false ;
    for(auto it : mp)
    {
        if(it.second%2 ==1)
        {
   b = true ;
   break;
        }
    }
    if(b)
    cout << "no" << endl;
    else
    cout << "yes" << endl;

 }
 void solve47()
 {
    int n,h;
    cin >> n;
    cin >> h;
    vector<int> ans;
  int k =0;
    for( int i=0;i<n;i++)
    {
       int x;
       cin >> x;
       ans.push_back(x) ;
       sort(ans.rbegin(),ans.rend());
       long long cost =0;
       for(int j=0;j<ans.size();j=j+2)
       {
     cost = cost + ans[j];
       }
       if(cost>h)
       {
        k=1;
        break;
       }
    }
    if(k)
    {
        cout << ans.size()-1<< endl;
    }
    else{
        cout << n << endl;
    }

 }
 void check(int num,unordered_map<int,int> &div)
 {
    
    int i =2;
    while(i*i<=num)
    {
        while(num%i == 0)
        {
            div[i]++;
            num = num/i;
        }
        i++;
    }
    if(num >1) div[num]++;

 }
 void solve48()
 {
    int n;
    cin >> n ;
    vector<int> a(n);
    for(int i=0;i<n;i++)
    {
        cin >> a[i];
    }
    unordered_map<int,int> divisor;
    for(int i=0;i<n;i++)
    {
        check(a[i],divisor);
    }
    bool b = false ;
    for(auto it : divisor)
    {
        if((it.second%n)!= 0)
        {
            b = true;
            break;
        }
    }
    if(b)
    cout << "NO" << endl;
    else
    cout << "YES" << endl;
 }
 void solve49()
 {
    int n;
    cin>> n ;
    vector<int> a(n);
    for(int i=0;i<n;i++)
    {
        cin >> a[i];

    }
    int j = 1;
    int count =0;
    long long int sum =0;
    for(int i=0;i<n;i++)
    {
        sum = sum+a[i];
        if(sum == j*j)
        {
            count++;
            j=j+2;
        }
        else if(sum > j*j)
        {
            while(sum > j*j)
            {
                j = j+2;
            }

            if(sum == j*j)
        {
            count++;
            j=j+2;
        }
        }
    }
    cout << count << endl;
 }
 void solve50()
 {
    int n ;
    cin >> n ;
    vector<int>b(n);
    vector<int>c(n);
    for(int i=0;i<n;i++)
    {
        cin >> b[i];
    }
    for(int i=0;i<n;i++)
    {
        cin >> c[i];
    }
    priority_queue<int> pq;
    long long ans =0;
    for(int i=0;i<n;i++)
    {
        int maxi = max(b[i],c[i]);
        int mini = min(b[i],c[i]);
        ans = ans + maxi;
        pq.push(mini);
    }
    ans = ans + pq.top();
    cout << ans << endl;
 }
int main() 
{
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
 
     int t;
     cin >> t;
     while(t--)
     {
   
     solve50();
     }
     
    
    
   
    
  
    
   
    
    
   
   
 
    return 0;
}






 
