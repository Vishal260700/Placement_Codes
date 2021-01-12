# Split the Array into subarrays of gcd property
arr = list(map(int, input().strip().split()))
# Eucledian
def getGCD(x, y):
    while(y):
        x, y = y, x%y
    return x
# O(N)
left = 0
right = len(arr) - 1
count = 0
while(right >= 0):
    for left in range(0, right + 1):
        if(getGCD(arr[left], arr[right]) > 1):
            right = left - 1
            count += 1
            break

print(count)

## Min Distance for buildings in a grid width and height and n number of buildings in total
def minDist(width, height, grid):
    res = 0
    for i in range(height):
        for j in range(width):
            temp = 2**31 - 1
            for n in grid:
                newtemp = abs(n.x - i) + abs(n.y - j)
                temp = min(temp, newtemp)
            res = max(temp, res)
    return res

class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def getMinDist(width, height, n):
    
    ans = 2**31 - 1
    total = width * height
    newGrid = []
    for i in range(0, total):
        xi = int(i/width)
        yi = int(i%width)
        newGrid.append(Cell(xi, yi))
        if(n == 1):
            ans = min(ans, minDist(width, height, newGrid))
        else:
            for j in range(i+1, total):
                xj = int(j/width)
                yj = int(j%width)
                newGrid.append(Cell(xj, yj))
                if(n == 2):
                    ans = min(ans, minDist(width, height, newGrid))
                else:
                    for k in range(j + 1, total):
                        xk = int(k/width)
                        yk = int(k%width)
                        newGrid.append(Cell(xk, yk))
                        if(n == 3):
                            ans = min(ans, minDist(width, height, newGrid))
                        else:
                            for w in range(k+1, total):
                                xw = int(w/width)
                                yw = int(w%width)
                                newGrid.append(Cell(xw, yw))
                                if(n==4):
                                    ans = min(ans, minDist(width, height, newGrid))
                                else:
                                    for z in range(w+1, total):
                                        xz = int(z/width)
                                        yz = int(z%width)
                                        newGrid.append(Cell(xz, yz))
                                        ans = min(ans, minDist(width, height, newGrid))
                                        newGrid.pop()
                                newGrid.pop()
                        newGrid.pop()
                newGrid.pop()
        newGrid.pop()
    
    return ans

w = int(input())
h = int(input())
n = int(input())
print(getMinDist(w, h, n))

# Check if a given number of very large length is divisible by 8 or not
# Input
number = input()
# If length is less than 3 i.e. 2 or 1.. simple check
if(len(number) < 3):
    Flag = False
    if(int(number) % 8 == 0):
        print(1)
        Flag = True
    number = ''.join(list(reversed(number)))
    if(int(number) % 8 == 0):
        print(1)
        Flag = True
    if(not Flag):
        print(0)
else:
    # Hashmap -- unique elements
    integers = {}
    for integer in number:
        if(integer not in integers):
            integers[integer] = 1
        else:
            integers[integer] += 1
    # check for unique elements in any triplet is divisible by 8
    for mulOf8 in range(104, 1000, 8):
        dup = mulOf8
        temp = {}
        while(dup):
            if(dup%10 not in temp):
                temp[dup%10] = 1
            else:
                temp[dup%10] += 1
            dup = dup//10
        Flag = True
        for key in temp.keys():
            if(key not in integers):
                continue
            if(temp[key] < integers[key]):
                Flag = False
        if(Flag):
            print(1)
            break
    if(not Flag):
        print(0)

# get positive integers who satisfy 1/x + 1/y = 1/N!  N given
N = int(input())
def fact(n):
    if(n == 0 or n == 1):
        return 1
    return fact(n-1) * n
# check for factors of (N^2) and (Y-N) is that factor
def getXY(N):
    count = 0
    n = fact(N)
    y = n+1
    while(y <= n*n + n):
        if((n*n) % (y-n) == 0):
            count += 1
        y += 1
    return count
print(getXY(N))

# Shared interest
# friends_weights, friends_from, friends_to

# dependencies
import itertools
from collections import defaultdict

def maxTokens(friends_nodes, friends_from, friends_to, friends_weight):
    # assigning set of friend_nodes to their shared weights
    #   Weights     nodes
    #     1         {1,2,3}  
    #     2         {1,2}
    #     3         {2,3,4}
    weights = defaultdict(set)
    for i in range(len(friends_from)):
        weights[friends_weight[i]].add(friends_from[i])
        weights[friends_weight[i]].add(friends_to[i])
    
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
            count[foo] += 1 
    
    for num in sorted(set(count.values()), reverse=True):
        pairs = [k for k,v in count.items() if v == num]
        if len(pairs) >= 2:
            return max([a*b for a, b in pairs])

friends_nodes = 4
friends_from = [1, 1, 2, 2, 2]
friends_to = [2, 2, 3, 3, 4]
friends_weight = [1, 2, 1, 3, 3 ]

print(maxTokens(friends_nodes, friends_from, friends_to, friends_weight))


# leaf to leaf -- max path sum
class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

parent = Node(5)
parent.left = Node(8)
parent.right = Node(10)
parent.left.left = Node(1)
parent.left.right = Node(3)
parent.right.left = Node(7)
parent.right.right = Node(9)
parent.left.left.left = Node(4)
parent.left.left.right = Node(5)
parent.left.right.left = Node(6)
parent.left.right.right = Node(11)
parent.right.left.left = Node(13)
parent.right.left.right = Node(14)
parent.right.right.right = Node(4)

res = [-2**31 + 1]
def solve(root):
    # Base Case
    if(root is None):
        return 0
    
    # Hypothesis
    left = solve(root.left)
    right = solve(root.right)
    
    # Recursive check
    res[0] = max(left + right + root.data, res[0])
    maxExcluded = max(left, right) + root.data
    return maxExcluded
solve(parent)
print(res[0])


# Min knight moves -- infinite grid bas no isValid
# make a set of visited tuple(x,y) for a bit efficency
class Cell:
    def __init__(self, x = 0, y = 0, moves = 0):
        self.x = x 
        self.y = y 
        self.moves = moves

def solve(sx, sy, tx, ty):
    
    queue = []
    queue.append(Cell(sx, sy, 0))
    
    while(queue):
        curr = queue.pop(0)
        if(curr.x == tx and curr.y == ty):
            return curr.moves
        dx = [1, 1, -1, -1, 2, -2, 2, -2]
        dy = [2, -2, 2, -2, 1, 1, -1, -1]
        
        for i in range(8):
            newX = curr.x + dx[i]
            newY = curr.y + dy[i]
            queue.append(Cell(newX, newY, curr.moves + 1))
    
print(solve(0, 0, 5, 5))

## Count Vowel Permutation -- do modulo
# DP
DP = [[-1 for x in range(6)] for y in range(n+1)]
def solve(n, lastElem):
    # Base Case
    if(n == 0):
        return 1
    
    # Memoization
    if(lastElem == 'a'):
        secIndex = 1
    elif(lastElem == 'e'):
        secIndex = 2
    elif(lastElem == 'i'):
        secIndex = 3
    elif(lastElem == 'o'):
        secIndex = 4
    elif(lastElem == 'u'):
        secIndex = 5
    else:
        secIndex = 0
    # Memoized Return
    if(DP[n][secIndex] != -1):
        return DP[n][secIndex]
    
    # recursive calls according to rules (Memoization)
    if(lastElem == 'a'):
        DP[n][secIndex] = solve(n-1, 'e') % (10**9 + 7)
        return DP[n][secIndex]
    elif(lastElem == 'e'):
        DP[n][secIndex] = (solve(n-1, 'a') + solve(n-1, 'i')) % (10**9 + 7)
        return DP[n][secIndex]
    elif(lastElem == 'i'):
        DP[n][secIndex] = (solve(n-1, 'a') + solve(n-1, 'e') + solve(n-1, 'o') + solve(n-1, 'u')) % (10**9 + 7)
        return DP[n][secIndex]
    elif(lastElem == 'o'):
        DP[n][secIndex] = (solve(n-1, 'i') + solve(n-1, 'u'))  % (10**9 + 7)
        return DP[n][secIndex]
    elif(lastElem == 'u'):
        DP[n][secIndex] = solve(n-1, 'a') % (10**9 + 7)
        return DP[n][secIndex]
    else:
        DP[n][secIndex] = (solve(n-1, 'a') + solve(n-1, 'e') + solve(n-1, 'i') + solve(n-1, 'o') + solve(n-1, 'u')) % (10**9 + 7)
        return DP[n][secIndex]
return solve(n, '')

# Recursive
def solve(n, lastElem):
    # Base Case
    if(n == 0):
        return 1
    # recursive calls according to rules
    if(lastElem == 'a'):
        return solve(n-1, 'e')
    elif(lastElem == 'e'):
        return solve(n-1, 'a') + solve(n-1, 'i')
    elif(lastElem == 'i'):
        return solve(n-1, 'a') + solve(n-1, 'e') + solve(n-1, 'o') + solve(n-1, 'u')
    elif(lastElem == 'o'):
        return solve(n-1, 'i') + solve(n-1, 'u')
    elif(lastElem == 'u'):
        return solve(n-1, 'a')
    else:
        return solve(n-1, 'a') + solve(n-1, 'e') + solve(n-1, 'i') + solve(n-1, 'o') + solve(n-1, 'u')
return solve(n, '')
                
                
            
            
            
        
