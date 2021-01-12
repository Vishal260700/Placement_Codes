## Min sum after K operations
import heapq
import math

Arr = list(map(int, input().strip().split()))
K = int(input())

Arr = [-x for x in Arr]

heapq.heapify(Arr)

while(K):
    maxEle = heapq.heappop(Arr)
    maxEle = -maxEle
    maxEle = math.ceil(maxEle/2)
    heapq.heappush(Arr, -maxEle)
    K -= 1

print(-sum(Arr))

## Inserting element based on identity and index
n = int(input())
index = list(map(int, input().strip().split()))
identity = list(map(int, input().strip().split()))

Arr = []
for i in range(0, len(index)):
    pos = index[i]
    ele = identity[i]
    
    if(pos == 0 and len(Arr) == 0):
        Arr.append(ele)
    elif(len(Arr) <= pos):
        Arr.append(ele)
    else:
        if(pos == 0):
            Arr = [ele] + Arr[:]
        else:# In between
            Arr = Arr[:pos] + [ele] + Arr[pos:]
print(Arr)

## Beautiful Substring vowels - aeiou
string = input()
parents = {'u' : 'o', 'o' : 'i', 'i' : 'e', 'e' : 'a', 'a' : ''}
# Linear O(N)
def linear(string):
    
    DP = {}
    isa = False
    ise = False
    isi = False
    iso = False
    isu = False
    
    for i in range(0, len(string)):
        if(string[i] == 'a'):
            isa = True
            if('a' not in DP):
                DP['a'] = 0
            DP['a'] = DP['a'] + 1
        elif(string[i] == 'e' and isa):
            ise = True
            if('e' not in DP):
                DP['e'] = 0
            DP['e'] = max(DP['a'] + 1, DP['e'] + 1)
        elif(string[i] == 'i' and ise):
            isi = True
            if('i' not in DP):
                DP['i'] = 0
            DP['i'] = max(DP['i'] + 1, DP['e'] + 1)
        elif(string[i] == 'o' and isi):
            iso = True
            if('o' not in DP):
                DP['o'] = 0
            DP['o'] = max(DP['i'] + 1, DP['o'] + 1)
        elif(string[i] == 'u' and iso):
            isu = True
            if('u' not in DP):
                DP['u'] = 0
            DP['u'] = max(DP['u'] + 1, DP['o'] + 1)
    
    if(isa and ise and isi and iso and isu):
        return DP['u']
    else:
        return 0
# Recursive
def solve(pointer, string, parent):
    
    if(pointer >= len(string)):
        return 0
    
    if(parent == string[pointer]):
        return (1 + solve(pointer + 1, string, parent))
    
    if(string[pointer] in parents):
        if(parents[string[pointer]] == parent):
            return max(solve(pointer + 1, string, parent), 1 + solve(pointer + 1, string, string[pointer]))
        else:
            return solve(pointer + 1, string, parent)
    else:
        return solve(pointer + 1, string, parent)
    

print(solve(0, string, ''))
print(linear(string))

## Sorted Arrangements
# Fenwick Tree - Insert elements with least cost
class BIT:
    
    def __init__(self, size):
        self.size = size
        self.Tree = [0 for x in range(0, self.size)]
    
    def update(self, index, value):
        while(index < self.size):
            self.Tree[index] += value
            index += index & (-index)
    
    def sum(self, index):
        total = 0
        while(index > 0):
            total += self.Tree[index]
            index -= index & (-index)
        return total
            

def solve(Arr):
    bitTree = BIT(10**2)
    res = 0
    for i in range(0, len(Arr)):
        bitTree.update(Arr[i], 1)
        larger = i + 1 - bitTree.sum(Arr[i])
        smaller = bitTree.sum(Arr[i] - 1)
        res += 2 * min(larger, smaller) + 1
    return res
    
Arr = list(map(int, input().strip().split()))
print(solve(Arr))

# we have to find the longest substring with utmost K 0s

binString = input()
K = int(input())

# O(N)
maxLen = 0
start = 0
zeroes = 0
length = 0
for i in range(0, len(binString)):
    if(binString[i] == '0'):
        zeroes += 1
        length += 1
        if(zeroes > K):
            for j in range(start, len(binString)):
                length -= 1
                if(binString[j] == '0'):
                    start = j + 1
                    zeroes -= 1
                    break
        maxLen = max(maxLen, length)
    else:
        length += 1

print(max(maxLen, length))

## Unique paths in a grid with obstacle
# Grid Formtion
grid = []
rows = int(input())
while(rows):
    row = list(map(int, input().strip().split()))
    grid.append(row)
    rows -= 1
# DP
DP = [[0 for x in range(0, len(grid[0]))] for y in range(0, len(grid))]
# Base Case
if(grid[0][0] == 0):
    DP[0][0] = 1
# Base Cases
for y in range(1, len(grid)):
    if(grid[y][0] == 0): # no obstacle
        DP[y][0] = DP[y-1][0]
for x in range(1, len(grid[0])):
    if(grid[0][x] == 0):
        DP[0][x] = DP[0][x-1]
# Tabulation
for y in range(1, len(grid)):
    for x in range(1, len(grid[0])):
        if(grid[y][x] == 0):
            DP[y][x] = DP[y-1][x] + DP[y][x-1]
print(DP[-1][-1])

        
## Max square in a grid
if(len(matrix) == 0):
    return 0
maxLen = 0
DP = [[0 for x in range(0, len(matrix[0]))] for y in range(0, len(matrix))]
for y in range(0, len(matrix)):
    for x in range(0, len(matrix[0])):
        if(matrix[y][x] == '1'):
            DP[y][x] = 1
            maxLen = max(maxLen, DP[y][x])

for y in range(1, len(matrix)):
    for x in range(1, len(matrix[0])):
        if(matrix[y][x] == '1'):
            DP[y][x] = min(DP[y-1][x], DP[y][x-1], DP[y-1][x-1]) + 1
            maxLen = max(maxLen, DP[y][x])
return maxLen**2

# Get min height of a tree with given inorder and level order traversal
inorder = list(map(int, input().strip().split()))
levelorder = list(map(int, input().strip().split()))

queue1 = []
queue2 = []
queue1.append(levelorder[0])
k = 1
height = 0

while(queue1 or queue2):
    
    if(queue1):
        height += 1
    while(queue1):
        curr = queue1.pop(0)
        for i in range(0, len(inorder)):
            if(inorder[i] == curr):
                break
        if(i > 0 and inorder[i-1] != -1 and k < len(inorder)):
            queue2.append(levelorder[k])
            k += 1
        if(i < len(inorder)-1 and inorder[i+1] != -1 and k < len(inorder)):
            queue2.append(levelorder[k])
            k += 1
        inorder[i] = -1
        
    if(queue2):
        height += 1
    while(queue2):
        curr = queue2.pop(0)
        for i in range(0, len(inorder)):
            if(inorder[i] == curr):
                break
        if(i > 0 and inorder[i-1] != -1 and k < len(inorder)):
            queue1.append(levelorder[k])
            k += 1
        if(i < len(inorder) -1 and inorder[i+1] != -1 and k < len(inorder)):
            queue1.append(levelorder[k])
            k += 1
        inorder[i] = -1

print(height)

# Get max beauty connected cities
n = int(input())
m = int(input())
maxTime = int(input())

n1 = n
beauty = []
while(n1):
    beauty.append(int(input()))
    n1 -= 1 

frm = []
m1 = m
while(m1):
    frm.append(int(input()))
    m1 -= 1

to = []
m2 = m
while(m2):
    to.append(int(input()))
    m2 -= 1

time = []
m3 = m 
while(m3):
    time.append(int(input()))
    m3 -= 1 

graph = {}
for i in range(0, len(time)):
    src = frm[i]
    dest = to[i]
    
    if(src not in graph):
        graph[src] = [[dest, time[i]]]
    else:
        graph[src].append([dest, time[i]])
    
    if(dest not in graph):
        graph[dest] = [[src ,time[i]]]
    else:
        graph[dest].append([src, time[i]])

def dfs(vertex, timeSoFar, beautyVal):
    # Base Case
    if(timeSoFar > maxTime):
        return 
    # we came back i.e. we either stop trip here or move forward for another adventure
    if(vertex == 0):
        ans[0] = max(ans[0], beautyVal)
    # Traversal
    if(vertex in graph):
        for neighbours in graph[vertex]:
            newVert = neighbours[0]
            newTime = neighbours[1]
            if(not visited[newVert]):
                visited[newVert] = True
                dfs(newVert, timeSoFar + newTime, beautyVal + beauty[newVert])
                visited[newVert] = False
            else:
                dfs(newVert, timeSoFar + newTime, beautyVal)
ans = [0]
visited = [False for x in range(n)]
dfs(0, 0, beauty[0])
print(ans[0])
    
    
    
## Triplets with sum less than equal to k 
arr.sort()
ans = 0
for i in range(len(arr) - 2):
    j = i + 1
    k = len(arr) - 1
    while(j < k):
        if(arr[i] + arr[j] + arr[k] >= sum):
            k -= 1
        else:
            ans += (k - j)
            j += 1
return ans

# Count Binary Substrings
ans = 0
prev = 0
curr = 1
for i in range(1, len(s)):
    if(s[i] != s[i-1]):
        ans += min(prev, curr)
        prev = curr
        curr = 1
    else:
        curr += 1
return ans + min(prev, curr)

## Shared interests
from collections import defaultdict

def maxTokens(friends_nodes, friends_from, friends_to, friends_weight):
    # assigning set of friend_nodes to their shared weights
    #   Wieghts      nodes
    #     1         {1,2,3}  
    #     2         {1,2}
    #     3         {2,3,4}
    weights = defaultdict(set)
    for i in range(len(friends_from)):
        weights[friends_weight[i]].add(friends_from[i])
        weights[friends_weight[i]].add(friends_to[i])
    # print(weights)
    # make set of pairs for each weight 
    #    Wieghts      nodes
    #      1         (1,2),(2,3),(1,3)  
    #      2         (1,2)
    #      3         (2,3),(3,4),(2,4)
    # count no of pairs 
    # {(1,2):2, (2,3): 2, (1,3):1, (3,4):1, (2,4):1}
    
    count = defaultdict(int)
    for key, val in weights.items():
        for foo in list(itertools.combinations(val, 2)):
            count[foo]+=1 
        
    # print(count)
    for num in sorted(set(count.values()), reverse=True):
        # print(num, )
        pairs = [k for k,v in count.items() if v == num]
        if len(pairs) >= 2:
            return max([a*b for a, b in pairs])
        

friends_nodes=4
friends_from =  [1, 1, 2, 2, 2]
friends_to =   [2, 2, 3, 3, 4]
friends_weight =  [1, 2, 1, 3, 3 ]

print(maxTokens(friends_nodes, friends_from, friends_to, friends_weight))


# Beautiful Arrangements ith pos devide i and i divided ith pos element
# Arranging numbers
res = [0]
visited = [False for x in range(N+1)]

def solve(arr):
    
    if(len(arr) == N):
        res[0] += 1
        return
    
    for i in range(1, N + 1):
        j = len(arr) + 1
        if(visited[i] == False and (i % j == 0 or j % i == 0)):
            
            arr.append(i)
            visited[i] = True
            
            solve(arr)
            
            # Back Track
            visited[i] = False
            arr.pop()
solve([])
return res[0]

# Min number of distinct numbers after removing M elemts 
# Python3 program for the above approach

# Function to return minimum distinct
# character after M removals
def distinctNumbers(arr, m, n):

	count = {}

	# Count the occurences of number
	# and store in count
	for i in range(n):
		count[arr[i]] = count.get(arr[i], 0) + 1

	# Count the occurences of the
	# frequencies
	fre_arr = [0] * (n + 1)
	for it in count:
		fre_arr[count[it]] += 1

	# Take answer as total unique numbers
	# and remove the frequency and
	# subtract the answer
	ans = len(count)

	for i in range(1, n + 1):
		temp = fre_arr[i]
		if (temp == 0):
			continue
			
		# Remove the minimum number
		# of times
		t = min(temp, m // i)
		ans -= t
		m -= i * t

	# Return the answer
	return ans

# Driver Code
if __name__ == '__main__':

	# Initialize array
	arr = [ 2, 4, 1, 5, 3, 5, 1, 3 ]

	# Size of array
	n = len(arr)
	m = 2

	# Function call
	print(distinctNumbers(arr, m, n))

## Find length of longest subsequence of one string which is substring of another string
# Python3 program to find maximum 
# length of subsequence of a string 
# X such it is substring in another 
# string Y. 

MAX = 1000

# Return the maximum size of 
# substring of X which is 
# substring in Y. 
def maxSubsequenceSubstring(x, y, n, m): 
	dp = [[0 for i in range(MAX)] 
			for i in range(MAX)] 
			
	# Initialize the dp[][] to 0. 

	# Calculating value for each element. 
	for i in range(1, m + 1): 
		for j in range(1, n + 1): 
			
			# If alphabet of string 
			# X and Y are equal make 
			# dp[i][j] = 1 + dp[i-1][j-1] 
			if(x[j - 1] == y[i - 1]): 
				dp[i][j] = 1 + dp[i - 1][j - 1] 

			# Else copy the previous value 
			# in the row i.e dp[i-1][j-1] 
			else: 
				dp[i][j] = dp[i][j - 1] 
				
	# Finding the maximum length 
	ans = 0
	for i in range(1, m + 1): 
		ans = max(ans, dp[i][n]) 
	return ans 

# Driver Code 
x = "ABCD"
y = "BACDBDCD"
n = len(x) 
m = len(y) 
print(maxSubsequenceSubstring(x, y, n, m)) 
    
# Make palindrome from substring
def canMakePaliQueries(self, s, queries):
    """
    :type s: str
    :type queries: List[List[int]]
    :rtype: List[bool]
    """
    
    class SubList:
        def __init__(self,l):
            self.st = l
        def __sub__(self,other):
            tmp = []
            for i in range(len(other.st)):
                tmp.append(self.st[i]-other.st[i])
            return SubList(tmp)

        def __add__(self,s):
            tmp = self.st[:]
            tmp[ord(s)-ord("a")] += 1
            return SubList(tmp)
    
    
    cur,preS = [0 for i in range(26)],[]
    for i in s:
        cur[ord(i)-ord("a")] += 1
        preS.append(SubList(cur[:]))
    ans = []
    for l,r,chance in queries:
        tmp = preS[r]-preS[l] + s[l]
        tmps = 0
        for i in tmp.st:
            tmps += i%2
        ans.append(tmps//2<=chance)
    return ans

