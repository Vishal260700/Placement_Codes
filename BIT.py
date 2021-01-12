## Codes for Stack and Queue

# Window Max

# O(n*k)
A = list(A)
res = []
for i in range(0, len(A) - B + 1):
    res.append(max(A[i:i+B]))
return res

# Heap - https://docs.python.org/3/library/heapq.html - O(nlogn)
# Use DS Heap - Max Heap
import heapq
# Min heap to Max heap
nums = [-num for num in nums]
# Result
res = []

for i in range(0, len(nums) - k + 1):
    heap = nums[i:i+k]
    heapq.heapify(heap)
    res.append(-heapq.heappop(heap))

return res

# Deque - Double ended Queue - O(n) and O(k)
# Unique DS - priority Que type
from collections import deque
deQue = deque()

# Store max element index and minElement index (may be present or not)
for i in range(0, k):
    while (deQue and nums[i] >= nums[deQue[-1]]):
        deQue.pop()
    deQue.append(i)

# result
res = []

for i in range(k, len(nums)):
    
    res.append(nums[deQue[0]])
    
    # remove elements on left of window
    while(deQue and deQue[0] <= i-k):
        deQue.popleft() # like pop(0)
    
    # same part as earlier
    while (deQue and (nums[i] >= nums[deQue[-1]])):
        deQue.pop()
    
    deQue.append(i)
# At the end we need final max element also
res.append(nums[deQue[0]])

return res

## Min platforms for train
## Greedy - O(nlogn)
arrival.sort()
departure.sort()

platforms = 1
result = 1
i = 1 # arrival
j = 0 # departure

while(i < len(arrival) and j < len(arrival)):
    
    if(arrival[i] <= departure[j]):
        platforms += 1
        i += 1
    elif(arrival[i] > departure[j]):
        platforms -= 1
        j += 1
    
    if(result < platforms):
        result = platforms

print(result)

## Map based - o(n) and O(2400 or 1)
result = 1

# time clock
graph = [0 for x in range(0, 2400)]

for i in range(0, len(arrival)):
    graph[arrival[i]] += 1
    
    graph[departure[i] + 1] -= 1 # usse ek aage wale ko bada de kyuki same time par aane par we need another platform as stated in questions

for i in range(0, 2400):
    
    graph[i] = graph[i] + graph[i-1]
    
    result = max(result, graph[i])

print(result)

## Next Greater Element -> 
# Input: nums1 = [4,1,2], nums2 = [1,3,4,2].
# Output: [-1,3,-1]
# O(n^2) -- Brute force
res = []
for i in range(0, len(nums1)):
    Flag = False
    Right = False
    for j in range(0, len(nums2)):
        if(nums1[i] == nums2[j]):
            Right = True
        if(nums2[j] > nums1[i] and Right):
            res.append(nums2[j])
            Flag = True
            break
    if(not Flag or not Right):
        res.append(-1)
return res

# O(n) worst case O(n^2) -- Optimized
nextGreatestElement = {} # key = value (key ka next greatest element)
stack = []

for i in range(0, len(nums2)):
    while(stack and stack[-1] < nums2[i]):
        temp = stack.pop() # new key jiska next element is nums2[i]
        nextGreatestElement[temp] = nums2[i]
    stack.append(nums2[i])

for i in range(0, len(nums1)):
    nums1[i] = nextGreatestElement[nums1[i]] if nums1[i] in nextGreatestElement else -1

return nums1

# Next smaller element problem statement
    Input 1:
        A = [4, 5, 2, 10, 8]
    Output 1:
        G = [-1, 4, -1, 2, 2]
    Explaination 1:
        index 1: No element less than 4 in left of 4, G[1] = -1
        index 2: A[1] is only element less than A[2], G[2] = A[1]
        index 3: No element less than 2 in left of 2, G[3] = -1
        index 4: A[3] is nearest element which is less than A[4], G[4] = A[3]
        index 4: A[3] is nearest element which is less than A[5], G[5] = A[3]
# O(n) worst case O(n^2)
# Basic Stack helper functions
def push(Arr, x):
    Arr.append(x)
    return Arr
def pop(Arr):
    Arr.pop()
    return Arr
def top(Arr):
    return Arr[-1]
def size(Arr):
    return len(Arr)
def isEmpty(Arr):
    return (len(Arr) == 0)
        
# given list of Integers - A
stack = []
minByFar = []

for i in range(0, len(A)):
    if(isEmpty(stack)):
        stack = push(stack, -1)
        if(isEmpty(minByFar)):
            minByFar = push(minByFar, A[i])
    elif(top(minByFar) < A[i]):
        stack = push(stack, top(minByFar))
        minByFar = push(minByFar, A[i])
    else:
        
        temp = minByFar
        while(len(temp) != 0):
            if(top(temp) < A[i]):
                stack = push(stack, top(temp))
                minByFar = push(minByFar, A[i])
                break
            temp = pop(temp)
        if(isEmpty(temp)):
            stack = push(stack, -1)
            minByFar = push(minByFar, A[i])

return stack

## Check paranthesis
stack = []
openBrackets = {'(': ')', '[' : ']', '{' : '}'}

for i in s:
    if(i in openBrackets):
        stack.append(i)
    else:
        if(len(stack) == 0):
            return False
        popped = stack.pop()
        if(i != openBrackets[popped]):
            return False

return (True if len(stack) == 0 else False)

## Min Stack
def __init__(self):
    """
    initialize your data structure here.
    """
    self.stack = []
    self.minimum = None
    

def push(self, x):
    """
    :type x: int
    :rtype: None
    """
    if(len(self.stack) == 0):
        self.stack.append(x)
        self.minimum = x
    elif(x < self.minimum):
        self.stack.append(2*x - self.minimum) # important part i.e. encoding the values
        self.minimum = x
    else:
        self.stack.append(x)
    

def pop(self):
    """
    :rtype: None
    """
    if(len(self.stack) == 0):
        return
    else:
        x = self.stack.pop()
        if(x < self.minimum):
            self.minimum = (2*self.minimum) - x

def top(self):
    """
    :rtype: int
    """
    if(len(self.stack) == 0):
        return
    else:
        temp = self.stack[-1]
        if(temp < self.minimum):
            return self.minimum
        else:
            return temp

def getMin(self):
    """
    :rtype: int
    """
    if(len(self.stack) == 0):
        return
    else:
        return self.minimum

## Majority Element >= N/2 times
# Optimized O(n) -- Moore Voting Algorithms
count = 0
curr = None

for num in nums:
    if(count == 0):
        curr = num
    count += (1 if curr == num else -1)
return curr


# No hashmap required or use linear space O(nlogn), O(1)
nums.sort()
curr = nums[0]
count = 1
for i in range(1, len(nums)):
    if(nums[i] == curr):
        count += 1
    else:
        count = 1
        curr = nums[i]
    if(count > len(nums)//2):
        return curr
return curr


# HashMap based O(N), O(N)
space = dict()
for i in range(0, len(nums)):
    if(nums[i] not in space):
        space[nums[i]] = 1
    else:
        space[nums[i]] += 1

for i in space.keys():
    if(space[i] > len(nums)//2):
        return i

# Majority Element > N/3 times
# Optimized O(n), O(1)
# Moore Voting Algorithm - 4 variables
curr1 = None
curr2 = None
count1 = 0
count2 = 0

for num in nums:
    if(curr1 == num):
        count1 += 1
    elif(curr2 == num):
        count2 += 1
    elif(count1 == 0):
        curr1 = num
        count1 += 1
    elif(count2 == 0):
        curr2 = num
        count2 += 1
    else:
        count1 -= 1
        count2 -= 1

res = []

for i in [curr1, curr2]:
    if(nums.count(i) > len(nums)//3):
        res.append(i)

return res


# Linear Space O(nlogn), O(1)
nums.sort()
curr = nums[0]
count = 1
res = []

for i in range(1, len(nums)):
    if(curr == nums[i]):
        count += 1
    else:
        if(count > len(nums)//3):
            res.append(curr)
        count = 1
        curr = nums[i]

if(count > len(nums)//3):
    res.append(curr)

return res

    

# Hashmap O(N), O(N)
space = dict()

for num in nums:
    if(num not in space):
        space[num] = 1
    else:
        space[num] += 1

res = []

for key in space.keys():
    if(space[key] > len(nums)//3):
        res.append(key)

return res

## Check in a 2d matrix
# Binary Search
if(len(matrix) == 0):
    return False
if(len(matrix[0]) == 0):
    return False

start = 0
end = len(matrix)*len(matrix[0])

if(target < matrix[0][0]):
    return False
if(target > matrix[-1][-1]):
    return False

def binarySearch(matrix, start, end, target):
    
    if(start > end):
        return False
    
    mid = (start+end)//2
    
    i = mid / len(matrix[0])
    j = mid % len(matrix[0])
    
    if(matrix[i][j] == target):
        return True
    elif(matrix[i][j] > target):
        return binarySearch(matrix,start, mid - 1, target)
    else:
        return binarySearch(matrix, mid + 1, end, target)
    
return binarySearch(matrix, start, end, target)

# Brute force
for i in range(0, len(matrix)):
    for j in range(0, len(matrix[i])):
        if(matrix[i][j] == target):
            return True
return False

## Power(X,n)
# Optimal O(log(N))
def powerHelper(number, power):
    
    if(power == 0):
        return 1
    
    temp = powerHelper(number, power//2)
    
    if(power % 2 == 0):
        return temp*temp
    else:
        return temp*temp*number

if(n >= 0):
    return powerHelper(x, n)

return 1/powerHelper(x, -n)

# Brute force - O(N)
if(n < 0):
    n = -n
    number = 1.0

    while(n != 0):
        number = x*number
        n -= 1

    return (1/number)
elif(n == 0):
    return 1
elif(n > 0):

    number = 1.0

    while(n != 0):
        number = x*number
        n -= 1

    return number

## Count distinct elements in a window
# Optimized - O(n) and O(k)
uniqElements = {}
uniques = K
for i in range(0, K):
    if(arr[i] not in uniqElements):
        uniqElements[arr[i]] = 1
    else:
        uniqElements[arr[i]] -= 1
        uniques -= 1

res = []
res.append(uniques)

for i in range(1, N - K + 1):
    if(arr[i-1] == arr[i+K-1]):
        res.append(uniques)
    else:
        # prev element
        if(arr[i-1] in uniqElements):
            if(uniqElements[arr[i-1]] == 1):
                uniques -= 1
                del uniqElements[arr[i-1]]
            else:
                uniqElements[arr[i-1]] += 1
        
        # new Elements
        if(arr[i+K-1] in uniqElements):
            uniqElements[arr[i+K-1]] -= 1
        else:
            uniques += 1
            uniqElements[arr[i+K-1]] = 1
        
        # Add to result
        res.append(uniques)
    
return res

# Kth largest element in an unsorted array
# Heap DS
# O(nlogn)
import heapq

max_heap = [-num for num in nums]

# function optimised for min heap but - sign helps to work for max heap
heapq.heapify(max_heap) 

while(k > 1):
    heapq.heappop(max_heap)
    k -= 1

return -heapq.heappop(max_heap)


# Approach - 1
# Sorting - O(nlogn)
nums.sort()
return nums[len(nums) - k]

## Get most significant bit ex- 17 -> 16 and 10 -> 8
# Brute force
def getMSB(n):
    
    i = 1
    while(i < n):
        i = i * 2
    
    if(i == n):
        return i
    else:
        return i/2
# Bit manipulation - O(1)
def getMSB(n):
    
    # right upto 32 bit structure
    n = n | n>>1
    n = n | n>>2
    n = n | n>>4
    n = n | n>>8
    n = n | n>>16 # just before 32 bit structure
    n = n + 1
    return n>>1

## Divide 2 integers without * or / operators
# get sign
if(dividend >= 0 and divisor > 0):
    sign = +1
elif(dividend <= 0 and divisor < 0):
    sign = +1
else:
    sign = -1
# taking in abs terms
dividend = abs(dividend)
divisor = abs(divisor)
# think of this process like getting MSB and decresasing dividend step by step from higher power
ans = 0
for power in range(31, -1, -1): # this range concept is imp
    if((divisor << power) <= dividend):
        ans += (1 << power)
        dividend -= (divisor << power)

ans = ans * sign
# if out of bonds
if(ans >= -2**31 and ans <= 2**31 - 1):
    return ans
else:
    return 2**31 - 1

# check if a number is power of 2 or not - O(1)
# less than equal to 0 wale NO
if(inputQuery > 0):
    # get most significant bit and compare with our value if equal then yes else no
    def MSB(n):
        
        n = n | n>>1
        n = n | n>>2
        n = n | n>>4
        n = n | n>>8
        n = n | n>>16
        n = n | n>>32 # here it was not specified of how much is the bit architecture so go upto 64 bits
        
        n = n + 1
        
        return n>>1
    if(MSB(inputQuery) == inputQuery):
        print('YES')
    else:
        print('NO')
else:
    print('NO')

## Validate Sudoku
# In check functions count refers to the original target value (that will be present once)
# True means valid else unvalid
def checkRow(target, rowNo, matrix):
    count = 0
    for col in range(0, len(matrix[0])):
        if(matrix[rowNo][col] == target and count == 1):
            return False
        elif(matrix[row][col] == target):
            count += 1
    return True
# True means valid else unvalid
def checkCol(target, colNo, matrix):
    count = 0
    for row in range(0, len(matrix)):
        if(matrix[row][colNo] == target and count == 1):
            return False
        elif(matrix[row][colNo] == target):
            count += 1
    return True
# check 3*3 matrix
def checkSubMatrix(target, tRow, tCol, matrix):
        count = 0
        # get submatrix section
        row_Section = tRow/3
        col_Section = tCol/3
        # Simple traversal
        for y in range(row_Section*3, row_Section*3 + 3):
            for x in range(col_Section*3, col_Section*3 + 3):
                if(matrix[y][x] == target and count == 1):
                    return False
                elif(matrix[y][x] == target):
                    count += 1
        return True

for row in range(0, len(matrix)):
    for col in range(0, len(matrix[0])):
        if(matrix[row][col]  == "."):
            continue
        s1 = checkRow(matrix[row][col], row, matrix)
        s2 = checkCol(matrix[row][col], col, matrix)
        s3 = checkSubMatrix(matrix[row][col], row, col, matrix)
        
        if( not ((s1 and s2) and s3)):
            return False

return True

## Solving Sudoku
class Solution(object):
    
    # True means valid else unvalid
    def checkRow(self, target, rowNo, matrix):
        for col in range(0, len(matrix[0])):
            if(matrix[rowNo][col] == target):
                return False
        return True

    # True means valid else unvalid
    def checkCol(self, target, colNo, matrix):
        for row in range(0, len(matrix)):
            if(matrix[row][colNo] == target):
                return False
        return True

    # check 3*3 matrix
    def checkSubMatrix(self, target, tRow, tCol, matrix):
            # get submatrix section
            row_Section = tRow/3
            col_Section = tCol/3
            # Simple traversal
            for y in range(row_Section*3, row_Section*3 + 3):
                for x in range(col_Section*3, col_Section*3 + 3):
                    if(matrix[y][x] == target):
                        return False
            return True
    
    # Overall Check
    def isValid(self, matrix, row, col, target):
        s1 = self.checkRow(target, row, matrix)
        s2 = self.checkCol(target, col, matrix)
        s3 = self.checkSubMatrix(target, row, col, matrix)
        
        if((s1 and s2) and s3):
            return True
        
        return False
    
    # Sudoku solver
    def sudokuSolver(self, matrix):
        for row in range(0, len(matrix)):
            for col in range(0, len(matrix[0])):
                if(matrix[row][col] == '.'):
                    for num in range(1, 10):
                        num = str(num)
                        
                        if(self.isValid(matrix, row, col, num)):
                            matrix[row][col] = num
                            
                            if(self.sudokuSolver(matrix)):
                                return True
                            
                            matrix[row][col] = '.'
                    return False
        return True
                        
    def solveSudoku(self, matrix):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
     
        return self.sudokuSolver(matrix)

# Longest Consequitive Sequence

# nlogn 
if(len(nums) == 0):
    return 0

nums.sort()
curr = nums[0]
count = 1
maxCount = 0
for i in range(1, len(nums)):
    if(curr + 1 == nums[i]):
        curr = nums[i]
        count += 1
    elif(curr == nums[i]):
        continue
    else:
        maxCount = max(count, maxCount)
        count = 1
        curr = nums[i]
return max(maxCount, count)

# n 
longestStreak = 0
nums = set(nums)

for num in nums:
    
    if(num - 1 not in nums):
        
        curr = num
        currStreak = 1
        
        while(curr + 1 in nums):
            currStreak += 1
            curr = curr + 1
        
        longestStreak = max(longestStreak, currStreak)

return longestStreak



# Add 2 numbers with BITs -- https://leetcode.com/problems/sum-of-two-integers/discuss/489210/Read-this-if-you-want-to-learn-about-masks
# Bit masking
mask = 0xffffffff
while(b & mask > 0):
    
    carry = (a & b)
    a = a ^ b
    b = (carry << 1)

return (a & mask) if b > 0 else a

## Largest Number
# Input: [3,30,34,5,9]
# Output: "9534330"
class LargerNum(str):
    def __lt__(x, y):
        return x + y > y + x
class Solution: # main Class
    def largestNumber(self, nums):
        largest_number = ''.join(sorted(map(str, nums), key = LargerNum))
        return '0' if largest_number[0] == '0' else largest_number