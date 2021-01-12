# Median of stream of integers

# Heaps - Max and Min

# Helper Func
def balanceHeaps(maxHeap, minHeap):
    while(abs(len(maxHeap) - len(minHeap)) > 1):
        if(len(maxHeap) > len(minHeap)):
            heapq.heappush(minHeap, -1*heapq.heappop(maxHeap))
        else:
            heapq.heappush(maxHeap, -1*heapq.heappop(minHeap))

import heapq # Min heap
maxHeap = []
minHeap = []
N = int(input())
while(N):
    temp = int(input())
    if(len(maxHeap) == 0 and len(minHeap) == 0):
        print(temp)
        heapq.heappush(maxHeap, -temp) # Max heap
    else:
        # Pushing
        if(temp < -maxHeap[0]):
            heapq.heappush(maxHeap, -temp)
        else:
            heapq.heappush(minHeap, temp)
        # Balancing
        balanceHeaps(maxHeap, minHeap)
        # Median Cal
        if(len(maxHeap) == len(minHeap)):
            print((-1*maxHeap[0] + minHeap[0])//2)
        elif(len(maxHeap) > len(minHeap)):
            print(-1 * maxHeap[0])
        elif(len(minHeap) > len(maxHeap)):
            print(minHeap[0])
    N -= 1

## Insertion Sort
def insert(Arr, x):
    if(len(Arr) == 0):
        Arr.append(x)
        return Arr
    else:
        if(x <= Arr[0]):
            res = [x]
            res.extend(Arr)
            return res
        elif(x >= Arr[-1]):
            Arr.append(x)
            return Arr
        else:
            res = []
            for i in range(0, len(Arr) - 1):
                if(x >= Arr[i] and x <= Arr[i+1]):
                    res.append(Arr[i])
                    res.append(x)
                    res.extend(Arr[i+1:])
                    return res
N = int(input())
arr = []
while(N):
    arr = insert(arr, int(input()))
    if(len(arr) % 2 == 0):
        t1 = arr[len(arr)//2]
        t2 = arr[len(arr)//2 - 1]
        print(int((t1 + t2)/2))
    else:
        print(arr[len(arr)//2])
        
    N -= 1

# Maximum SubArray Sum possible by replacing an array element by its square
Arr = list(map(int, input().strip().split()))
# Tabulation
DP = [ [-1, -1] for i in range(0, len(Arr)) ]
# Base Cases
DP[0][0] = Arr[0]
DP[0][1] = Arr[0]**2
max_sum = max(DP[0][0], DP[0][1])
# Tabulating
for i in range(1, len(Arr)):
    DP[i][0] = max(DP[i-1][0] + Arr[i], Arr[i]) # 0th Index represent no squaring just adding and starting new
    DP[i][1] = max([ DP[i-1][1] + Arr[i], Arr[i], DP[i-1][0] + Arr[i]**2 ]) # 1st Index represent squaring and adding to prev, new start, prev squared and add present elem
    max_sum = max([ max_sum, DP[i][0], DP[i][1] ])
print(max_sum)

## Interesting Primes, check for numbers less than equal to N which can be represented as a^2 + b^4 where a and b are > 0
testCases = int(input())
def checkPrime(n):
    if(n <= 1):
        return False
    if(n <= 3):
        return True
    if(n%2 == 0 or n%3 == 0):
        return False
    i = 5
    while(i**2 <= n):
        if(n%i == 0 or n%(i+2) == 0):
            return False
        i += 6
    return True
    
while(testCases):
    N = int(input())
    count = 0
    primes = set()
    for i in range(1, int(N**0.5) + 1):
        for j in range(1, int(N**0.25) + 1):
            temp = i**2 + j**4
            if(temp > N):
                break
            else:
                if(checkPrime(temp) and temp not in primes):
                    count += 1
                    primes.add(temp)
    print(count)
    testCases -= 1 

## Find Primes - K from 1 to y - x + 1 and min p primes
def checkPrime(n):
    if(n <= 1):
        return False
    if(n <= 3):
        return True
    if(n%2 == 0 or n%3 == 0):
        return False
    i = 5
    while(i*i <= n):
        if(n%i == 0 or n%(i+2) == 0):
            return False
    return True

testCases = int(input())
while(testCases):
    xyp = list(map(int, input().strip().split()))
    x = xyp[0]
    y = xyp[1]
    p = xyp[2]
    
    Flag = False
    count = 0
    for i in range(x, y + 1):
        if(checkPrime(i)):
            count += 1
        if(count == p):
            print(i)
            Flag = True
            break
    if(not Flag):
        print(-1)
    testCases -= 1

## Lego

## DP -- TLE (Some Test Cases) - check Lego Hackerank
## n -> Wall Height
## m -> Wall Width
mod = 10**9 + 7
n = n % mod
m = m % mod

# Height is 1 and varyingWidth
constHeight = []
constHeight.append(0) # No width
constHeight.append(1) # No Height
if(m > 1):
    constHeight.append(2)
if(m > 2):
    constHeight.append(4)
if(m > 3):
    constHeight.append(8)
if(m > 4):
    for i in range(5, m + 1):
        constHeight.append( (constHeight[i-1] + constHeight[i-2] + constHeight[i-3] + constHeight[i-4] ) % mod )

# VaryingHeight = constHeight ** height
varyingHeight = []
for i in range(0, len(constHeight)):
    varyingHeight.append((constHeight[i] ** n) % mod)

res = [0] * (m+1)
res[0] = 0
res[1] = 1
for i in range(2, m+1):
    res[i] = varyingHeight[i]
    for j in range(1, i):
        res[i] = (res[i] - res[j] * varyingHeight[i - j]) % mod
return res[-1] % mod

# Fastest
mod = 10**9 + 7
# Indexs are height
varyingHeights = [ [1] * 1001, [1] * 1001, [1] * 1001, [1] * 1001, [1] * 1001, [1] * 1001, [1] * 1001, [1] * 1001, [1] * 1001, [1] * 1001 ]
# One Height
varyingHeights[0][2] = 2
varyingHeights[0][3] = 4
# Building up const height
for i in range(4, 1001):
    varyingHeights[0][i] = (varyingHeights[0][i-1] + varyingHeights[0][i-2] + varyingHeights[0][i-3] + varyingHeights[0][i-4]) % mod
# Building up next heights based on the prev heights
for i in range(1, 1001):
    varyingHeights[1][i] = (varyingHeights[0][i] ** 2) % mod
    varyingHeights[2][i] = (varyingHeights[1][i] ** 2) % mod
    varyingHeights[3][i] = (varyingHeights[2][i] ** 2) % mod
    varyingHeights[4][i] = (varyingHeights[3][i] ** 2) % mod
    varyingHeights[5][i] = (varyingHeights[4][i] ** 2) % mod
    varyingHeights[6][i] = (varyingHeights[5][i] ** 2) % mod
    varyingHeights[7][i] = (varyingHeights[6][i] ** 2) % mod
    varyingHeights[8][i] = (varyingHeights[7][i] ** 2) % mod
    varyingHeights[9][i] = (varyingHeights[8][i] ** 2) % mod

an = [1] * 1001
for i in range(1, m+1):
    for j in range(10):
        if ((n >> j) & 1):
            an[i] = an[i] * varyingHeights[j][i]
    an[i] = an[i] % mod

res = [1] * 1001
for i in range(2, m+1):
    s = 0
    for j in range(1, i):
        s = s + res[j] * an[i - j]
    res[i]  = (an[i] - s) % mod
return res[m]

## Divide a binary array into 3 equal decimal parts
BinArr = list(map(int, input().strip().split()))

# total number of ones
ones = sum(BinArr)

if(ones == 0):
    print(0)
elif(ones % 3 != 0):
    print(-1)
else:
    
    k = ones//3
    count = 0
    intervals = []
    for i in range(0, len(BinArr)):
        if(BinArr[i] == 1):
            count += 1
            if(count in {1, k + 1, 2*k + 1}):
                intervals.append(i)
            if(count in {k, 2*k, 3*k}):
                intervals.append(i)
    i1, j1, i2, j2, i3, j3 = intervals
    
    if(not(BinArr[i1: j1 + 1] == BinArr[i2: j2 + 1] == BinArr[i3: j3 + 1])):
        print(-1)
    else:
        trailingZeroes = len(BinArr) - j3 - 1
        
        zeroesbetween12 = i2 - j1 - 1
        zeroesbetween23 = i3 - j2 - 1
        if(zeroesbetween12 < trailingZeroes or zeroesbetween23 < trailingZeroes):
            print(-1)
        else:
            j1 = j1 + trailingZeroes
            j2 = j2 + trailingZeroes
            print('Points of Cuts', j1, j2 + 1)
            
## String Game - Delete K grouped same charecters from a string
string = input()
K = int(input())
stack = []
numberStack = []
for i in range(0, len(string)):
    if(len(stack) == 0):
        stack.append(string[i])
        numberStack.append(1)
    else:
        if(stack[-1] == string[i]):
            numberStack[-1] += 1
            if(numberStack[-1] == K):
                numberStack.pop()
                stack.pop()
        else:
            stack.append(string[i])
            numberStack.append(1)
print(stack, numberStack)    

## Minimise Summation - (Bi - Bj)^2 for i and j from 1 to k
# Helper Func
def calFunc(arr):
    res = 0
    for i in range(len(arr)):
        for j in range(len(arr)):
            res += (arr[i] - arr[j])**2
    return res

# Sort First
# K size window calculation
Arr = list(map(int, input().strip().split()))
K = int(input())

# First Step
Arr.sort()

# Second Step
minX = 2**31 - 1
i = 0
while(i < len(Arr) - K + 1):
    minX = min(minX, calFunc(Arr[i:i+K]))
    i += 1
print(minX)

## Dijkstra Algorithm -- https://www.youtube.com/watch?v=ba4YGd7S-TY (Very good explanation) -- O((V+E)log(V)) - Adjacency List
## Wont work with negative edges (use bellman ford)
## Dijsktra with Adjacency List -> O(ElogV) <- O((E+V)logV) but E >= V
## It is said to implement it with priority queue but python does not have we need import but below is basic with no such need
## Path and min distance both calculated until we find the answer
## Complexity - O(V^2) and if Adjacency list used - O(E log V)
    
    # Graph formed (Adj Matrix) -- used adj matrix, if using adj list only traversal method to find neighbour changes else logic is same
    testCases = int(input())
    graph = []
    while(testCases > 0):
        vertices = int(input())
        while(vertices > 0):
            vertices -= 1
            temp = list(map(int, input().strip().split()))
            graph.append(temp)
        testCases -= 1

    # Target settings
    temp = list(map(int, input().strip().split()))
    start = temp[0]
    end = temp[1]

    # Variables
    import sys
    parents = [None for x in range(0, len(graph))] # will help to build path

    # State management
    visited = [0 for x in range(0, len(graph))]
    minDistance = [sys.maxsize for x in range(0, len(graph))]
    minDistance[start] = 0

    # Algorithm design
    # extract min minDistance 
    # extracted ka vertex mil gaya 
    # iss vertex ke saare neighbours ke liye
    # min distance nikal and compare with their minDistance array value if kam then update
    # after above check add them to a visited array
    # update its parent too
    # do above till end is visited

    # return the minValue one's index (ExtractMin in Heaps)
    def getMin(arr, visited):
        minSoFar = sys.maxsize
        minIndex = -1
        for i in range(0, len(arr)):
            if((minSoFar > arr[i]) and (visited[i] == 0)):
                minSoFar = arr[i]
                minIndex = i
        return minIndex

    # Looping through till either end is reached or we run out of visiting nodes (psuedo ends)
    for i in range(0, len(visited)):
        if((i == end) and (visited[i] != 0)):
            # stop here
            break
        else:
            # get neighbours
            minIndex = getMin(minDistance, visited)
            visited[minIndex] = 1
            current = minIndex

            for i in range(0, len(graph)):
                ## Very Important this part
                if(((graph[current][i] > 0) and(visited[i] == 0)) and (minDistance[i] > minDistance[current] + graph[current][i])):
                    minDistance[i] = minDistance[current] + graph[current][i]
                    parents[i] = current
    
    # Answering
    print(minDistance[end])
    finalResult = []

    while(end > 0):
        finalResult.append(end)
        end = parents[end]
    finalResult.append(0)

    print(list(reversed(finalResult)))

# Maximum overlapping subInterval - Intervals and Value per interval get max value of intervals
5
1 6 3
4 7 4
10 11 10
12 25 6
20 24 5
numberOfIntervals = int(input()) # given
# Keep track of max Time and value
maxVal = -2**31 + 1
index = -1
Dict = {}
while(numberOfIntervals):
    interval = list(map(int, input().strip().split()))
    for i in range(interval[0], interval[1] + 1):
        if(i not in Dict):
            Dict[i] = interval[2]
        else:
            Dict[i] += interval[2]
        if(Dict[i] > maxVal):
            maxVal = Dict[i]
            index = i
    numberOfIntervals -= 1
print(maxVal, index)

# Number of Islands based on 8-connectivity Traversal
grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"],
  ["0","0","0","1","1"]
]

# Helper Func
def isInside(x, y):
    if((x >= 0 and x < len(grid[0])) and (y >= 0 and y < len(grid))):
        return True
    return False
# Traversing
def dfs(x, y):
    grid[y][x] = '0'
    dx = [1, 1, 1, -1, -1, -1, 0, 0]
    dy = [0, 1, -1, 0, 1, -1, 1, -1]
    for i in range(8):
        newX = x + dx[i]
        newY = y + dy[i]
        if(isInside(newX, newY) and grid[newY][newX] == '1'):
            dfs(newX, newY)
# Starter Code
count = 0
for y in range(0, len(grid)):
    for x in range(0, len(grid[0])):
        if(grid[y][x] == '1'):
            count += 1
            dfs(x, y)
print(count)

# check if 2 strings are permutations of each other -- Lower chars only
string1 = input()
string2 = input()

set1 = [0] * 26
set2 = [0] * 26

for char in string1:
    set1[ord(char) - 97] += 1
for char in string2:
    set2[ord(char) - 97] += 1

if(set1 == set2):
    print(1)
else:
    print(0)

## Two Costs A and B for x+-1 and x//2
# I am considering both Cost A and B as 1
X = int(input())
def solve(X):
    # Base Case
    if(X == 0):
        return 0
    if(X == 1):
        return 1
    # Conditional (given)    
    if(X%2 == 0):
        return solve(X//2) + 1
    else:
        return 1 + min(solve(X - 1), solve(X + 1))
print(solve(X))

## Coins of Demonations 3,5,10 find ways to get sum N
testCases = int(input())

while(testCases):
    
    N = int(input())
    
    DP = [0 for x in range(0, N + 1)]
    DP[0] = 1
    
    for i in range(3, N + 1):
        DP[i] += DP[i-3]
    
    for i in range(5, N + 1):
        DP[i] += DP[i-5]
    
    for i in range(10, N + 1):
        DP[i] += DP[i - 10]
    
    print(DP[-1])
    
    testCases -= 1

## St bits equal to X in range L to R ( 1 indexed )
def getSetBits(N):
    count = 0
    while(N):
        N = N & (N-1)
        count += 1
    return count

Arr = list(map(int, input().strip().split()))
L = int(input())
R = int(input())
X = int(input())

maxForX = -2**31 + 1
for i in range(L-1, R):
    if(getSetBits(Arr[i]) == X):
        maxForX = max(maxForX, Arr[i])
print(maxForX)    

// C++ Solution

#include<bits/stdc++.h>
using namespace std;
#define ll long long 
#define llin(n) scanf("%lld",&n)
#define llin2(n,m) scanf("%lld %lld",&n, &m)
#define llin3(n,m,k) scanf("%lld %lld %lld",&n, &m, &k)
#define llin4(n,m,j,k) scanf("%lld %lld %lld %lld",&n, &m, &j ,&k)
ll st[2000005], arr[2000000]={0}, brr[2000000]={0};
vector < ll > vec;
string str;
ll mod=1000000007;
map< ll, ll> mp;
stack < ll > stk;
ll fac(ll n)
{
	ll i, prod=1;
	for(i=1;i<=n;i++)
		prod=((i%mod)*(prod%mod))%mod;
	return prod;
}
ll power(ll x, ll y, ll p)
{
	ll res = 1;      
	x = x % p;   
	while (y > 0)
	{
		if (y & 1)
			res = (res*x) % p;
		y = y>>1; // y = y/2
		x = (x*x) % p;  
	}
	return res;
}
ll gcd(ll a, ll b)
{
	if(a==0)
		return b;
	return gcd(b%a,a);
}
ll countSetBits(ll n) 
{ 
	ll count1 = 0; 
	while (n) 
	{ 
		count1 += n & 1; 
		n >>= 1; 
	} 
	return count1; 
} 
ll construct(ll l, ll r, ll ind)
{
	if(l>=r)
	{
		st[ind]=countSetBits(arr[l]);
		return st[ind];
	}
	ll mid=(l+r)/2;
	st[ind]=construct(l,mid,2*ind+1) + construct(mid+1,r,2*ind+2);
	return st[ind];
}
ll query(ll si, ll ei, ll ql, ll qr, ll ind)
{
	if(si>qr || ei<ql)
		return 0;
	if(si>=ql && ei<=qr)
		return st[ind];
	ll mid=(si+ei)/2;
	return query(si,mid,ql,qr,2*ind+1) + query(mid+1,ei,ql,qr,2*ind+2);
}
int main()
{
	ll x, y, i, j, k, n, m, num, temp, fn, cnt=0, d, l, r, flag=0, sum1=0, a, b, len1=0, t, mx=LLONG_MIN, mn=LLONG_MAX, ind;
	cin >> n;
	for(i=0;i<n;i++)
		cin >> arr[i];
	construct(0,n-1,0);
	cin >> num;
	for(i=0;i<num;i++)
	{
		cin >> l >> r;
		printf("%lld\n",query(0,n-1,l,r,0));
	}
	return 0;
}

# Min moves to reach from point named 2 to one of the edge filled with 0s and 1s (block)

# helper Function
def isValid(x, y, grid, visited):
    if( (x >= 0 and x < len(grid[0])) and (y >= 0 and y < len(grid)) ):
        if(visited[y][x] == 0 and grid[y][x] == 0): # unvisited and plant (path possible)
            return True
    return False

def bfs(grid):
    
    # get start pos
    for i in range(0, len(grid)):
        for j in range(0, len(grid[0])):
            if(grid[i][j] == 2):
                sx = j
                sy = i
    
    visited = [[0 for x in range(0, len(grid[0]))] for y in range(0, len(grid))]
    
    from collections import deque
    q = deque()
    q.append([sy, sx, 0])
    
    while(q):
        curr = q.popleft()
        currY = curr[0]
        currX = curr[1]
        currMoves = curr[2]
        
        # Set to visited
        visited[currY][currX] = 1
        
        # Edge Check
        if((currY == 0 or currY == len(grid) - 1) or (currX == 0 or currX == len(grid[0]) - 1)):
            print(currMoves)    
            break
        
        # Four Directions
        # UP
        if(isValid(currX, currY - 1, grid, visited)):
            q.append([currY - 1, currX, currMoves + 1])
        # DOWN
        if(isValid(currX, currY + 1, grid, visited)):
            q.append([currY + 1, currX, currMoves + 1])
        # LEFT
        if(isValid(currX - 1, currY, grid, visited)):
            q.append([currY, currX - 1, currMoves + 1])
        # RIGHT
        if(isValid(currX + 1, currY, grid, visited)):
            q.append([currY, currX + 1, currMoves + 1])

graph = [[1, 1, 1, 0, 1 ], 
       [1, 0, 2, 0, 1 ], 
       [0, 0, 1, 0, 1 ], 
       [1, 0, 1, 1, 0 ] ]
bfs(graph)


# Buildings of heights side by side given get max rect area (Histograms can also be given instead of bar graphs)
# height is array 
if not height:
    return 0
stack = []
maxrect = 0
for i in range(len(height)):
    curr = height[i]
    while stack:
        if stack[-1][1] > curr:
            (index, h, left) = stack.pop()
            maxrect = max(h*(i-index-1)+left, maxrect)
        elif stack[-1][1] == curr:
            stack.pop()
        else:
            break
    if stack:
        stack.append((i, curr, curr*(i-stack[-1][0])))
    else:
        stack.append((i, curr, curr*(i+1)))
while stack:
    (index,h,left) = stack.pop()
    maxrect = max(h*(len(height)-index-1)+left, maxrect)
return maxrect         

####################################################################################################
This question had an obscure function whose value we had to calculate, the parameters of the function were such that any direct application would cross the integer limits… sorry cant remember any specifics.
I will simply this question for you as far as I remember.
You are given n,m,a,b,c as inputs. You need to calculate a function.
f(i) = (pow(i,i) + A*pow(i,5) + B + C*(something))*f(i-1), i>=2
Given, f(1) = 1 and
A = m*a;
B = A+A*(b+c)
C = A+B+A*(Math.ceil(log(base 10)(a+b+c)))
You need to find f(n)%m
1 <= n <= 10^18
1 <= m <= 10^6
1 <= a,b,c <= 10^12

####################################################################################################
Solution: So, the function basically boils down to f(i) = pow(i,i)*f(i-1).
So, if(n>=m) just return 0;
Else, calculate till f(n)%m in which case n<=10^6. So, no TLE I suppose.

You are given a function f(i) = f(i-1) * (A*i9 + (B*i! + 1)*ii + C*(i^(i^i))). Where A = a*m, B = A*(b+c), C = 5*B + (A*log10(b*c))
Input:- n, m, a, b, c
Output:- f(n)%m
Solution- In case (n >= m) it’s divisible by m so return 0;
In case(n < m) the problem boils down to (n^n * (n-1)^(n-1) * ….. * 1^1)%m as A, B and C are divisible by m.

# Max size subset with given sum
set[] = {2, 3, 5, 7, 10, 15},
         sum  = 10
Output : 3
The largest sized subset with sum 10
is {2, 3, 5}

def isSubsetSum(arr, n, sum):
	
	# The value of subset[i][j] will 
	# be true if there is a subset of 
	# set[0..j-1] with sum equal to i 
	subset = [[False for x in range(n + 1)]
					for y in range (sum + 1)]
	count = [[0 for x in range (n + 1)]
				for y in range (sum + 1)]

	# If sum is 0, then answer is true 
	for i in range (n + 1):
		subset[0][i] = True
		count[0][i] = 0
	
	# If sum is not 0 and set is empty, 
	# then answer is false 
	for i in range (1, sum + 1):
		subset[i][0] = False
		count[i][0] = -1
		

	# Fill the subset table in bottom up manner 
	for i in range (1, sum + 1): 
		for j in range (1, n + 1):
			subset[i][j] = subset[i][j - 1]
			count[i][j] = count[i][j - 1]
			if (i >= arr[j - 1]) :
				subset[i][j] = (subset[i][j] or
								subset[i - arr[j - 1]][j - 1])

				if (subset[i][j]):
					count[i][j] = (max(count[i][j - 1], 
									count[i - arr[j - 1]][j - 1] + 1))
	return count[sum][n]

# Driver code
if __name__ == "__main__":

	arr = [2, 3, 5, 10]
	sum = 20
	n = 4
	print (isSubsetSum(arr, n, sum))

# Bar Graph Question -- remove exactly K bars
#include <bits/stdc++.h>
using namespace std;

// typedefs

// structure or class definitions

// constants

// function declarations
int bar_graph(vector<int> &v, int k);

// global variables

// main function
int main()
{

    int tc;
    int n, k;
    int i, j;

    cin >> tc;
    while (tc--)
    {
        cin >> n >> k;
        vector<int> v(n);
        for (i = 0; i < n; i++)
            cin >> v[i];
        cout << bar_graph(v, k) << '\n';
    }
    return 0;
}

int bar_graph(vector<int> &v1, int k)
{
    int i, n = v1.size();
    vector<int> v2;
    vector<int> dv;

    for (i = 1; i < n; i++)
        v2.push_back(v1[i] - v1[i - 1]);

    n--;
    k--;
    deque<ll> dq(n - k);
    for (i = 0; i < n - k; i++)
    {
        while (!dq.empty() && v2[i] >= v2[dq.back()])
            dq.pop_back();
        dq.push_back(i);
    } // at the end of this loop dq will hold index of max element of 1st sub-array of size n-k

    for (; i < n; i++)
    {
        // element at the front of the queue is the index of largest element of previous window
        dv.push_back(v2[dq.front()]);
        // cout << v2[dq.front()] << " ";

        // remove the indexes which are out of this window
        while (!dq.empty() && dq.front() <= i - n + k)
            dq.pop_front();

        while (!dq.empty() && v2[i] >= v2[dq.back()])
            dq.pop_back();

        dq.push_back(i);
    }

    // index of the maximum element of last window
    dv.push_back(v2[dq.front()]);
    // cout << v2[dq.front()];

    // int n1 = dv.size();
    int mx = INT_MIN;
    for (i = 0; i < n - k; i++)
        if (dv[i] > mx)
            mx = dv[i];

    return mx;
}

## Shortest Path with K cards for no toll
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int maxn = 1e5 + 14, maxk = 20;
int n, m, k;
ll d[maxn][maxk];
struct E{
    int u, w;
};
vector<E> g[maxn];
int main(){
    ios::sync_with_stdio(0), cin.tie(0);
    cin >> n >> m >> k;
    for(int i = 0; i < m; i++){
        int v, u, w;
        cin >> v >> u >> w;
        v--, u--;
        g[v].push_back({u, w});
        g[u].push_back({v, w});
    }
    memset(d, 63, sizeof d);
    d[0][0] = 0;
    set<pair<ll, pair<int, int> > > q({{0, {0, 0}}});
    while(q.size()){
        auto [v, used] = q.begin() -> second;
        q.erase(q.begin());
        auto upd = [&q](int u, int used, ll nw){
            if(used <= k && nw < d[u][used]){
                q.erase({d[u][used], {u, used}});
                d[u][used] = nw;
                q.insert({d[u][used], {u, used}});
            }
        };
        for(auto [u, w] : g[v]){
            upd(u, used, d[v][used] + w);
            upd(u, used + 1, d[v][used]);
        }
    }
    for(int i = 0; i < n; i++)
        cout << *min_element(d[i], d[i + 1]) << ' ';
    cout << '\n';   
}