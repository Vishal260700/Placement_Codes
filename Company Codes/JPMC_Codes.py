
## Sum of Leaf Nodes at Min Level
Input : 
         1
        /  \
       2    3
     /  \     \
    4    5     8 
  /  \ 
 7    2      
Output :
sum = 5 + 8 = 13

Dict = {} # key is level and values are leaf nodes only at that level
level = 0 # start level is 0 and increases as we go down

def getMinLeafSum(root, level):
    
    if(root is None):
        return
    elif(root.left is None and root.right is None):
        if(level in Dict):
            Dict[level] += root.data
        else:
            Dict[level] = root.data
    else:
        getMinLeafSum(root.left, level + 1)
        getMinLeafSum(root.right, level + 1)

getMinLeafSum(root, level)

if(len(Dict) == 0):
    return 0

minLevel = min(Dict.keys())
return Dict[minLevel]

##  Sum Root to Leaf Numbers

Input: [4,9,0,5,1]
    4
   / \
  9   0
 / \
5   1
Output: 1026
Explanation:
The root-to-leaf path 4->9->5 represents the number 495.
The root-to-leaf path 4->9->1 represents the number 491.
The root-to-leaf path 4->0 represents the number 40.
Therefore, sum = 495 + 491 + 40 = 1026.

def formNumber(root, string):
    if(root is None):
        return 0
    
    if(root.left is None and root.right is None):
        return int(string + str(root.val))
    
    L = formNumber(root.left, string + str(root.val))
    R = formNumber(root.right, string + str(root.val))
    
    return int(L + R)

return formNumber(root, '')

## Range of numbers in BST in range L to R
def traversal(root):
    if(root):
        if(root.val >= L and root.val <= R):
            self.ans += root.val
            traversal(root.left)
            traversal(root.right)
        if(root.val < L):
            traversal(root.right)
        if(root.val > R):
            traversal(root.left)
self.ans = 0
traversal(root)
return self.ans 

## Number of Subtrees in a BST where all elements lie in range L to R

def traversal(root):
    # Base Case
    if(root is None):
        return True
    # Hypothesis
    L = traversal(root.left)
    R = traversal(root.right)
    # If lie in range or not and its a complete subtree
    if((L and R) and (root.data >= l and root.data <= h)):
        count[0] += 1
        return True
    # else all
    return False

# this helps in state management in python
count = [0]

traversal(root)

return count[0]

## Number of BSTs in a BT

INT_MIN = -2**31
INT_MAX = 2**31

def num_BST(root):
    # Base
    if(root is None):
        return 0, INT_MIN, INT_MAX, True
    # Leaf
    if(root.left is None and root.right is None):
        return 1, root.data, root.data, True
    
    # Hypothesis
    L = num_BST(root.left)
    R = num_BST(root.right)
    
    # Hypothesis related variables
    lower = max(root.data, min(L[1], R[1]))
    higher = min(root.data, max(L[2], R[2]))
    
    # Checking if root value lies in left and right frontier andn also if its a complete subtree or not
    if((L[3] and R[3]) and (root.data >= L[1] and root.data <= R[2])):
        return 1 + L[0], R[0], lower, higher, True
    else:
        return L[0] + R[0], lower, higher, True

## Root to leaf all paths

def dfs(root, path):
            
    # Base
    if(root is None):
        return
    # Leaf
    if(not root.left and not root.right):
        if(path == ''):
            return path + str(root.val)
        else:
            return path + '->' + str(root.val)
    
    # Hypothesis
    if(path == ''):
        leftPath = dfs(root.left, path + str(root.val))
        if(leftPath):
            res.append(leftPath)
        rightPath = dfs(root.right, path + str(root.val))
        if(rightPath):
            res.append(rightPath)
    else:
        leftPath = dfs(root.left, path + '->' + str(root.val))
        if(leftPath):
            res.append(leftPath)
        rightPath = dfs(root.right, path + '->' + str(root.val))
        if(rightPath):
            res.append(rightPath)

res = []
result = dfs(root, '')
if(result):
    res.append(str(result))
return res

## PAth Sum -- check if a root to leaf path sum equals given sum
def dfs(root, total):
            
    if(root is None):
        return False
    
    if(root.left is None and root.right is None):
        if(total + root.val == sum):
            return True
        else:
            return False
    
    Left = dfs(root.left, total + root.val)
    Right = dfs(root.right, total + root.val)
    
    return Left or Right

return dfs(root, 0)

## Path sum 2 - total how many such paths (all paths in array)
def dfs(root, path, total):
            
    if(root is None):
        return False, []
    
    if(root.left is None and root.right is None):
        if(total + root.val == sum):
            return True, path + [root.val]
        else:
            return False, []
    
    Left = dfs(root.left, path + [root.val], total + root.val)
    Right = dfs(root.right, path + [root.val], total + root.val)
    
    if(Left and Left[0]):
        res.append(Left[1])
    
    if(Right and Right[0]):
        res.append(Right[1])
    
res = []
result = dfs(root, [], 0)
if(result and result[0]):
    res.append(result[1])
return res

# make leaf nodes in a LL (rightmost to Leftmost)
        
class ListNode(object):
    def __init__ (self, val):
        self.val = val
        self.next = None
# Iterative
queue = []
queue.append(root)

head = ListNode(None)
res = head

while(queue):
    curr = queue.pop()
    if(curr is None):
        continue
    if(curr.left is None and curr.right is None):
        res.next = ListNode(curr.val)
        res = res.next
    else:
        queue.append(curr.left)
        queue.append(curr.right)

head = head.next
ans = []
while(head):
    ans.append(head.val)
    head = head.next
return ans

# get the maximum left node (left of a node)
def traversal(root, Flag):
    
    # Base Case
    if(root is None):
        return 0
    
    # Left and Right Hypo
    leftMaxLeftNode = traversal(root.left, True)
    rightMaxLeftNode = traversal(root.right, False)
                
    if(Flag):
        return max([leftMaxLeftNode, rightMaxLeftNode, root.val])
    else:
        return max(leftMaxLeftNode, rightMaxLeftNode)

print(traversal(root, False))

# get deepest left leaf node
def traversal(root, Flag, level, maxLevel):
            
    if(root is None):
        return 0, 0
    
    if(root.left is None and root.right is None):
        if(Flag):
            if(maxLevel[0] < level):
                return level, root.val
            else:
                return maxLevel
        else:
            return maxLevel
    
    leftMaxLevel = traversal(root.left, True, level + 1, maxLevel)
    rightMaxLevel = traversal(root.right, False, level + 1, maxLevel)
    
    if(leftMaxLevel[0] > rightMaxLevel[0]):
        return leftMaxLevel
    else:
        return rightMaxLevel

print(traversal(root, False, 0, [0, 0]))

# Kth largest element in a BST
stack = []
while(True):
    while(root):
        stack.append(root)
        root = root.right
    root = stack.pop()
    k -= 1
    if(k == 0):
        return root.data
    root = root.left

https://www.spoj.com/problems/AGGRCOW/cstart=70
https://www.geeksforgeeks.org/place-k-elements-such-that-minimum-distance-is-maximized/
https://www.geeksforgeeks.org/find-possible-words-phone-digits/


## Find minimum possible size of array with given rules for removing elements

k = int(input())
Arr = list(map(int, input().strip().split()))

# Memoized
DP = [[-1 for x in range(0, len(Arr) + 1)] for y in range(0, len(Arr) + 1)]

def solve(arr, low, high, k):
    # Base Case i.e. not enough elems
    if(high - low + 1 < 3):
        return high - low + 1
    
    if(DP[low][high] != -1):
        return DP[low][high]
    
    # Consider low indexed in not in triplet
    res = 1 + solve(arr, low + 1, high, k)
    
    # Consider low indexed in in triplet
    
    # check for all possible other 2 points
    for i in range(low + 1, len(arr) - 1):
        for j in range(i + 1, len(arr)):
            if(arr[i] - arr[low] == k and arr[j] - arr[i] == k and solve(arr, low+1, i-1, k) == 0 and solve(arr, i+1, j-1, k) == 0):
                res = min(res, solve(arr, j + 1, high, k))
    DP[low][high] = res
    return res

print(solve(Arr, 0, len(Arr) - 1, k))

## Collect maximum points in a grid

grid = []
rows = int(input())
while(rows):
    temp = list(map(int, input().strip().split()))
    grid.append(temp)
    rows -= 1

# Helper func
def isValid(y, x1, x2, arr):
    if(y >= len(arr) or y < 0 or x1 < 0 or x1 >= len(arr[0]) or x2 < 0 or x2 >= len(arr[0])):
        return False
    return True

# Memoization
DP = [[[-1 for x2 in range(0, len(grid[0]) + 1)] for x1 in range(0, len(grid[0]) + 1)] for y in range(0, len(grid) + 1)]

# solve
def solve(arr, y, x1, x2):
    # Bingo
    if(y == len(arr) - 1 and x1 == 0 and x2 == len(arr[0]) - 1):
        return arr[y][x1] + arr[y][x2]
    
    # Check if point is valid or not
    if(isValid(y, x1, x2, arr) == False):
        return -2**31 + 1
    
    # Memoized
    if(DP[y][x1][x2] != -1):
        return DP[y][x1][x2]
    
    # Common point
    if(x1 == x2):
        res = arr[y][x1]
    else:
        res = arr[y][x1] + arr[y][x2]
    
    ## 9 situations
    # x1 constant
    one = solve(arr, y + 1, x1, x2 + 1)
    two = solve(arr, y + 1, x1, x2 - 1)
    three = solve(arr, y + 1, x1, x2)
    # x1 + 1
    four = solve(arr, y + 1, x1 + 1, x2 + 1)
    five = solve(arr, y + 1, x1 + 1, x2 - 1)
    six = solve(arr, y + 1, x1 + 1, x2)
    # x1 - 1
    seven = solve(arr, y + 1, x1 - 1, x2 + 1)
    eight = solve(arr, y + 1, x1 - 1, x2 - 1)
    nine = solve(arr, y + 1, x1 - 1, x2)
    
    DP[y][x1][x2] = res + max([one, two, three, four, five, six, seven, eight, nine])
    return DP[y][x1][x2]

print(solve(grid, 0, 0, len(grid[0]) - 1))

# Check if a tree is subtree of given tree
# Helper Func
def isSame(root1, root2):
    
    if(root1 is None and root2 is None):
        return True
    
    if(root1 is None or root2 is None):
        return False
    
    if(root1.data != root2.data):
        return False
    
    return (isSame(root1.left, root2.left) and isSame(root1.right, root2.right))

def isSubTree(T1, T2):
    # Code here
    
    if(T2 is None):
        return True
    
    if(T1 is None):
        return False
    
    if(isSame(T1, T2)):
        return True
    
    return (isSubTree(T1.left, T2) or isSubTree(T1.right, T2))

# Vertex Cover Problem
class Node:
    def __init__(self, value):
        self.data = value
        self.left = None
        self.right = None
        # DP
        self.cvr = -1

def solve(root):
    
    # Base Cases
    if(root is None):
        return 0 
    if(root.left is None and root.right is None):
        return 0
      
    # DP 
    if(root.cvr != -1):
        return root.cvr
    
    # curr root is included
    inc = 1 + solve(root.left) + solve(root.right)
    
    # curr root excluded
    exc = 0
    if(root.left):
        exc += 1 + solve(root.left.left) + solve(root.left.right)
    if(root.right):
        exc += 1 + solve(root.right.left) + solve(root.right.right)
    
    root.cvr = min(exc, inc)
    return root.cvr
    
# form tree
root = Node(20)
# Left part
root.left = Node(8)
root.left.left = Node(4)
root.left.right = Node(12)
root.left.right.left = Node(10)
root.left.right.right = Node(14)
# Right part
root.right = Node(22)
root.right.right = Node(25)

print('Size of smallest vertex cover is ', solve(root))

## Weighted Job Schedule -- given 3 arrays startTime, endTime, profit

# Gawd Solution
# O(nlogn)
events = []
for i, (start, end) in enumerate(zip(startTime, endTime)):
    events.append((start, i+1))
    events.append((end, -(i+1)))

best = 0 # max profit so far
for _, index in sorted(events):
    if(index > 0):
        # start point
        profit[index - 1] += best # profit arr given
    else:
        # end point
        best = max(profit[-index - 1], best)
return best

# Sort based on end time -- imp step
allInfo = [[profit[i], startTime[i], endTime[i]] for i in range(0, len(startTime))]
allInfo.sort(key = lambda x : x[2])

# DP - O(n^2)
DP = [{} for i in range(0, len(allInfo) + 1)]

def solve(arr, pointer, parentEndTime):
    # Base Case
    if(pointer >= len(arr)):
        return 0
    
    # DP Check
    if(parentEndTime in DP[pointer]):
        return DP[pointer][parentEndTime]
    
    if(arr[pointer][1] < parentEndTime):
        return solve(arr, pointer + 1, parentEndTime)
    
    DP[pointer][parentEndTime] = max(solve(arr, pointer + 1, parentEndTime), arr[pointer][0] + solve(arr, pointer + 1, arr[pointer][2]))
    return DP[pointer][parentEndTime] 

return solve(allInfo, 0, -1)      
        
# Sort based on end time -- imp step
allInfo = [[profit[i], startTime[i], endTime[i]] for i in range(0, len(startTime))]
allInfo.sort(key = lambda x : x[2])

# Recursive O(n^n)
def solve(arr, pointer, parentEndTime):
    # Base Case
    if(pointer >= len(arr)):
        return 0
    
    if(arr[pointer][1] < parentEndTime):
        return solve(arr, pointer + 1, parentEndTime)
    
    return max(solve(arr, pointer + 1, parentEndTime), arr[pointer][0] + solve(arr, pointer + 1, arr[pointer][2]))

return solve(allInfo, 0, -1)
    


    
    
        