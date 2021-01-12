## Number of subsets mean
## https://practice.geeksforgeeks.org/problems/number-of-subsets-and-mean1225/1
minMean = min(Arr)
maxMean = max(Arr)

minCounts = 0
maxCounts = 0

for i in Arr:
    if(i == minMean):
        minCounts += 1
    if(i == maxMean):
        maxCounts += 1

return (2**maxCounts - 1 if(maxCounts > 1) else 1 , 2**minCounts - 1 if(minCounts > 1) else 1)








## Reverse a Number O(log(N)), O(1)
## If we do by string conversion and reversing speed will decrease
    if(x == 0):
        return x
    elif(x > 0):
        res = 0
        while(x != 0):
            rem = x%10
            x = x/10
            res = res*10 + rem
        return (res if res <= 2**31 - 1 else 0)
    else:
        x = -x
        res = 0
        while(x != 0):
            rem = x%10
            x = x/10
            res = res*10 + rem
        return (-res if (-res) >= -2**31 else 0)

## Reverse String - 2 cases of even/odd number of charecters O(N) swapping takes complexity
    
    for i in range(0, len(s)//2 + 1):
        if(i < len(s) - 1 - i):
            s[i], s[len(s) - 1 - i] = s[len(s) - 1 - i], s[i]

## Remove duplicates from sorted

## Two pointers
# Two pointers
    prev = 0
    prevValue = None
    
    for i in range(0, len(nums)):
        if(nums[i] != prevValue):
            prevValue = nums[i]
            nums[prev] = nums[i]
            prev += 1
    
    print (nums[:prev])

## Popping method
    i = 0
    while(i < len(nums)-1):
        if(nums[i] == nums[i+1]):
            nums.pop(i)
        else:
            i += 1
    
    return i+1

## First Unique charecter in a string
    # O(N) and O(1) as 26 charecters
    chars = {}
    deleted = set()

    for i in range(0, len(s)):
        if((s[i] not in chars) and (s[i] not in deleted)):
            chars[s[i]] = i
        elif(s[i] in chars):
            del chars[s[i]]
            deleted.add(s[i])

    if(len(chars) == 0):
        return -1
    else:
        return (min(values for values in chars.values()))

## Power of 3
    rem = 0
    while(n > 1):
        rem = n%3
        n = n/3
        if(rem != 0):
            return False
    if(n == 1):
        return True

## Move Zeroes
    size = len(nums)
    i = 0
    count = 0
    while(i < size):
        if(count > size):
            break
        if(nums[i] == 0):
            nums.pop(i)
            nums.append(0)
            count += 1
        else:
            i += 1
            count += 1

## Prime numbers less than n

# Approach 1 - O(n^2)
        if(n <= 2):
            return 0
        elif(n == 3):
            return 1
        else:
            count = 0
            for i in range(2, n):
                Flag = True
                for j in range(2, i//2 + 1):
                    if(i%j == 0):
                        Flag = False
                        break
                if(Flag):
                    count += 1
            return count

# Approach 2 - O(n^2) - slightly optimized
        if(n <= 2):
            return 0
        elif(n == 3):
            return 1
        else:
            count = 0
            for i in range(2, n):
                Flag = True
                for j in range(2, int(math.sqrt(i)) + 1):
                    if(i%j == 0):
                        Flag = False
                        break
                if(Flag):
                    count += 1
            return count
# Optimal - O(N^1.5)
        def isPrime(n):
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
                i = i + 6
            return True
        
        count = 0
        for i in range(2, n):
            if(isPrime(i)):
                count += 1
        return count

# Sieve Of Eratosthenes work for number less than 10 million O(nlog(log(n)))
        
        if(n <= 1):
            return 0
        
        prime = [True for i in range(n)]
        
        p = 2
        for i in range(2, len(prime)):    
            if(prime[i]):
                for j in range(p**2, n, p):
                    prime[j] = False
            p += 1
        
        count = 0
        for i in prime:
            if (i):
                count += 1
        
        return count - 2

# Total number of 1 bits
        # using the original number
        def bits(n):
            if(n <= 1):
                return n
            else:
                i = 2
                while(i <= n):
                    i = i*2

                if(i == n):
                    return 1
                else:
                    i = i/2
                    return 1 + bits(n - i)
        return bits(n)
        
        # using conversion O(1) as 32 bits only long
        n = str(bin(n))
        count = 0
        for char in n:
            if(char == '1'):
                count += 1
        return count

        # Bit masking - O(1) -- important concept
        mask = 1
        count = 0
        for i in range(0, 32):
            
            if(n & mask):
                count += 1
            
            mask = mask << 1
        
        return count

## Identify single occuring element

        # Using Dict - O(N), O(N)
        data = {}
        for num in nums:
            if(num not in data):
                data[num] = 1
            else:
                del data[num]
        for key in data.keys():
            return key
        
        # Without extra space - Sorting O(nlogn)
        nums.sort()
        i = 0
        while(i < len(nums) - 1):
            if(nums[i] == nums[i+1]):
                i += 1
            else:
                return nums[i]
            i += 1
        return nums[-1]

        # Bit manipulation - XOR
        res = nums[0]
        
        for i in range(1, len(nums)):
            res = res ^ nums[i]
        
        return res

## Valid Palindromic String - O(n)

    # First pass
    ans = []
    for i in s:
        if((ord(i) >= 97 and ord(i) <= 122)):
            ans.append(ord(i) - 32)
        elif((ord(i) >= 65 and ord(i) <= 90)):
            ans.append(ord(i))
        elif((ord(i) >= 48 and ord(i) <= 57)):
            ans.append(ord(i))
    # Second Pass
    for i in range(0, len(ans)//2):
        if(ans[i] != ans[len(ans) - 1- i]):
            return False
    return True
    
## Missing Number from (o to n) in a given array only 1 is missing

# O(N) and O(N)
    data = {}
    
    for num in nums:
        data[num] = 1
    
    for i in range(0, len(nums) + 1):
        if(i not in data):
            return i
# O(N) and O(1)
    expSum = 0
    realSum = 0
    for i in range(0, len(nums)):
        realSum += nums[i]
        expSum += i
    expSum += len(nums)
    return (expSum - realSum)

## Trailing Zeroes

        # Recursively JEE Style - O(logn)
        def powerOfx(n, x):
            count = 0
            temp = x
            while(n >= x):
                count += int(n/x)
                x = x*temp
            return count
        return min(powerOfx(n, 2), powerOfx(n, 5))

## Rotate Array - Shift all elements by k to right

        # Brute force - O(n) and O(n)
        res = [None for x in range(0, len(nums))]
        
        for i in range(0, len(nums)):
            if(i+k < len(nums)):
                res[i+k] = nums[i]
            elif(i + k >= len(nums)):
                res[i+k-len(nums)] = nums[i]
        
        print(res)

        # Inplace
        # Reversing method - O(n) and O(1)
        def reverse(start, end, arr):
            while(start < end):
                arr[start], arr[end] = arr[end], arr[start]
                start += 1
                end -= 1
        
        k = k % (len(nums)) # important line -- takes care of cases where k > len(nums)
        reverse(0, len(nums) - 1, nums)
        reverse(0, k - 1, nums)
        reverse(k, len(nums) - 1, nums)

## reverse Bits

        # O(n) and O(1) as 32 bits only
        temp = list(str(bin(n)))
        temp = temp[2:]
        
        # Convert to 32 bits - '{:032b}'.format(100) --> gives  32 bit structure of 100
        if(len(temp) < 32):
            for i in range(0, 32 - len(temp)):
                temp = ['0'] + temp
        
        for i in range(0, len(temp)//2):
            temp[i], temp[len(temp) - 1 - i] = temp[len(temp) - 1 - i], temp[i]
        
        binary = ''.join(temp)
        return int(binary, 2)

        # without conversion to string -- Working ?? O(1) and O(1)
        res = 0
        power = 31
        
        while(n):
            res += (n&1) << power
            n = n >> 1
            power -= 1
        
        return res

## Excel Sheet Column number "A" - 1 or "ZY" - 701
        
        # Simple math - O(n) and O(1)
        s = list(s)
        res = 0
        for i in range(0, len(s)):
            res += (26**(len(s) - 1 - i))*(ord(s[i]) - 64)
        return res


## Min Stack Design

class MinStack(object):

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
            self.stack.append(2*x - self.minimum)
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

## Checkk if it contains duplicates

        # Without extra space - O(nlogn) and O(1)
        nums.sort()
        for i in range(0, len(nums) - 1):
            if(nums[i] == nums[i+1]):
                return True
        return False
        
        # Hashmap - O(N), O(N)
        data = set()
        for num in nums:
            if(num not in data):
                data.add(num)
            else:
                return True
        return False


## Valid anagram - s = "anagram", t = "nagaram" -- True

        # Hashmap - 2 pass O(n) and O(n)
        Dict = {}
        for i in s:
            if(i not in Dict):
                Dict[i] = 1
            else:
                Dict[i] += 1
        for i in t:
            if(i in Dict):
                Dict[i] -= 1
                if(Dict[i] == 0):
                    del Dict[i]
            else:
                return False
        if(len(Dict)):
            return False
        else:
            return True

# Stock Buy sell, any number of buying and selling

        # Single Pass O(n)
        profit = 0
        
        for i in range(0, len(prices) - 1):
            if(prices[i] < prices[i+1]):
                profit += prices[i+1] - prices[i]
        
        return profit

# sqrt(x)

        # Brute force - O(n)
        if(x == 0 or x == 1):
            return x
        
        for i in range(1, x+1):
            if(i**2 == x):
                return i
            elif(i**2 > x):
                return (i-1)
        
        # Optimized O(n)
        ans = x//2
        
        while(ans*ans != x):
            if(ans*ans > x):
                ans = ans//2
            elif(ans*ans < x and (ans+1)*(ans+1) > x):
                return ans
            else:
                ans += 1
        
        return ans

        # Newtons method
        if(x == 0 or x == 1):
            return x
        else:
            approx = 0.5*x
            root = 0.5 * (approx + x/approx) # Newtons method
            
            while(root != approx):
                approx = root
                root = 0.5 * (approx + x/approx)
            
            return int(approx)
            
        
        
        


## Sum of left leaves
# Left side wali saari leaves ka sum nikal
# https://leetcode.com/problems/sum-of-left-leaves/submissions/
def leftSum(root, leftFlag):

    if(root is None):
        return 0

    if(root.left is None and root.right is None):
        if(leftFlag):
            return root.val
        else:
            return 0
    
    return leftSum(root.left, True) + leftSum(root.right, False)

return leftSum(root, False)

## Third maximum number
# three pointers 
import sys
# Three pointers
maxFirst = -sys.maxsize + 1
maxSecond = -sys.maxsize + 1
maxThird = -sys.maxsize + 1

for num in nums:
    if(num >= maxFirst):
        if(num != maxFirst):
            maxFirst, maxSecond, maxThird = num, maxFirst, maxSecond
    elif(num >= maxSecond):
        if(maxSecond != num):
            maxSecond, maxThird = num, maxSecond
    elif(num > maxThird):
        print(num)
        maxThird = num

if(maxThird == -sys.maxsize + 1):
    return maxFirst
else:
    return maxThird

# Convert to base of 7
if(num < 0):
    return '-' + self.convertToBase7(-num)

if(num == 0):
    return '0'

import math
MSB = int(math.log(num, 7))
res = ''

while(MSB >= 0):
    if(num/(7**MSB) >= 1):
        res += str(num/(7**MSB))
    else:
        res += '0'
    num = num%(7**MSB)
    MSB -= 1

return res

# Binary Search O(logN)
low = 0
high = len(nums) - 1

while(low <= high):
    
    mid = (low + high)//2
    
    if(nums[mid] == target):
        return mid
    elif(nums[mid] > target):
        high = mid - 1
    else:
        low = mid + 1

return -1

## Get length of lastword of a string
lastWord = ""
for i in range(len(s)-1, -1, -1):
    if(s[i] != ' '):
        lastWord = s[i] + lastWord
    else:
        if(len(lastWord) != 0):
            return len(lastWord)

return len(lastWord)

## remove duplicates from sorted Linked lists
if(head is None):
    return head
curr = head
while(curr.next):
    if(curr.val == curr.next.val):
        curr.next = curr.next.next
    else:
        curr = curr.next

return head

## Convert a LL of binary representation to Decimal
## Approach 1 - Two traversal O(N) and O(N)
# reverse LL
def reverseLL(head):

    prev = None
    curr = head

    while(curr):
        nextCurr = curr.next
        curr.next = prev
        prev, curr = curr, nextCurr
    
    return prev


revLL = reverseLL(head)

# Simple traversal
count = 0
res = 0
while(revLL):
    res += (2**count)*revLL.val
    count += 1
    revLL = revLL.next
return res

# Approach 2 - Single Traversal O(N) and O(1)
res = 0
while(head):
    res = res*2 + head.val
    head = head.next
return res

# Approach 3 - Bit manipulation O(N) and O(1) -- Fastest
res = head.val
while(head.next):
    res = (res << 1) | head.next.val # Bit level OR implementation
    head = head.next
return res

# Remove elements of target value in a Linked list
if(head is None):
    return head

curr = head
while(curr.next):
    if(curr.next.val == val):
        curr.next = curr.next.next
    else:
        curr = curr.next

if(head.val == val):
    return head.next
else:
    return head


# search Insert position
# Input: [1,3,5,6], 5
# Output: 2
low = 0
high = len(nums) - 1

while(low <= high):
    
    mid = (low + high)//2
    
    if(nums[mid] == target):
        return mid
    elif(nums[mid] > target):
        high = mid - 1
    elif(nums[mid] < target):
        low = mid + 1

return max(low, high)

## Valid Perfect Square -- check if a number is a perfect square
# with lib
return True if math.sqrt(num)%int(math.sqrt(num)) == 0 else False
# no lib -- Binary Search
low = 0
high = num

while(low <= high):
    
    mid = (low + high)//2
    
    squareVal = mid*mid
    
    if(squareVal == num):
        return True
    elif(squareVal > num):
        high = mid - 1
    else:
        low = mid + 1

return False

# Intersection of 2 arrays
# Input: nums1 = [1,2,2,1], nums2 = [2,2]
# Output: [2]
# Brute force - O(n + m) and O(n + m)
def set_intersection(set1, set2):
    return [x for x in set1 if x in set2]

nums1 = set(nums1)
nums2 = set(nums2)

if(len(nums1) > len(nums2)):
    return set_intersection(nums2, nums1)
else:
    return set_intersection(nums1, nums2)

# Built In Set Unions - O(n+m) and O(n+m)
nums1 = set(nums1)
nums2 = set(nums2)
return list(nums1 & nums2)

## Find Smallest Letter Greater Than Target -- Leetcode Question check for input and output
# Binary Search - O(logN) and O(1)
low = 0
high = len(letters) - 1
while(low <= high):
    mid = (low + high)//2
    
    if(letters[mid] == target):
        if(mid + 1 >= len(letters)):
            return letters[0]
        else:
            if(letters[mid + 1] == target):
                low = mid + 1
            else:
                return letters[mid + 1]
    elif(ord(letters[mid]) > ord(target)):
        high = mid - 1
    else:
        low = mid + 1

if(max(low, high) >= len(letters)):
    return letters[0]
else:
    return letters[max(low, high)]
        
## Climbing Stairs
# Input: 2
# Output: 2
# Explanation: There are two ways to climb to the top.
# 1. 1 step + 1 step
# 2. 2 steps

# Recursive
# Two choice 1 or 2
# n-1 or n-2
if(n == 0):
    return 1

if(n < 0):
    return 0

return (self.climbStairs(n-1)) + (self.climbStairs(n-2))

## DP
if(n <= 2):
    return n

DP = [-1 for x in range(0, n)]
DP[0] = 1
DP[1] = 2

for i in range(2, n):
    DP[i] = DP[i-1] + DP[i-2]

return DP[n-1]

## Count Negative Numbers in a Sorted Matrix
# Linear Search O(N^2)
count = 0
for y in range(0, len(grid)):
    for x in range(0, len(grid[y])):
        if(grid[y][x] < 0):
            count += 1
return count

# Binary Search O(nlogn)
res = 0
for grid in grids:
    
    low = 0
    high = len(grid)
    
    if grid[0] < 0:
        res += high
    elif grid[high-1] >= 0:
        continue
    else:
        while low < high:
            mid = (low+high)//2
            if grid[mid] < 0:
                high = mid
            elif grid[mid] >= 0:
                low = mid+1
        res += len(grid)-high
return res

# Shortest Unsorted Continuous Subarray - https://leetcode.com/problems/shortest-unsorted-continuous-subarray/solution/
# Input: [2, 6, 4, 8, 10, 9, 15]
# Output: 5
# O(nlogn), more better solutions in above link
sortedNums = list(sorted(nums))
        
if(len(nums) <= 1):
    return 0

start = 0
end = len(nums) - 1

startPart = -1
endPart = -1

while(start <= end):
    
    if(startPart != -1 and endPart != -1):
        return end - start + 1
    
    if(startPart == -1):
        if(nums[start] != sortedNums[start]):
            startPart = start
        else:
            start += 1  
    if(endPart == -1):
        if(nums[end] != sortedNums[end]):
            endPart = end
        else:
            end -= 1

return 0

## Find All Numbers Disappeared in an Array -- Good Question -- leetcode
# Input: [4,3,2,7,8,2,3,1]
# Output: [5,6]
# O(N) and O(N)
data = dict()
        
for i in range(1, len(nums)+1):
    data[i] = 1

for num in nums:
    if(num in data):
        del data[num]

return data.keys()

# O(N) and O(1) -- ??
length = len(nums)
  
for i in range(0, length):
    value = abs(nums[i]) - 1
    nums[value] = abs(nums[value])*(-1)

res = []
for i in range(0, length):
    if(nums[i] > 0):
        res.append(i+1)
return res

# House Robber
# Recursion
def maxRob(nums, pointer):
            
    # Base Case
    if(pointer >= len(nums)):
        return 0
    
    # Pointed Selected, Pointed not Selected
    return max(nums[pointer] + maxRob(nums, pointer + 2), maxRob(nums, pointer+1))

return maxRob(nums, 0)

# DP
DP = [0 for x in range(0, len(nums))]
        
if(len(nums) <= 0):
    return 0

if(len(nums) == 1):
    return nums[0]

if(len(nums) == 2):
    return max(nums)

DP[0] = nums[0]
DP[1] = max(nums[0], nums[1])

for i in range(2, len(nums)):
    DP[i] = max(DP[i-1], DP[i-2] + nums[i])

return DP[-1]

## Merge sorted Array
# O(N) and O(N)
res = []
        
pointer1 = 0
pointer2 = 0

while(pointer1 < len(nums1) and pointer2 < len(nums2)):
    if(nums1[pointer1] > nums2[pointer2]):
        res.append(nums1[pointer1])
        pointer1 += 1
    else:
        res.append(nums2[pointer2])
        pointer2 += 1

while(pointer1 < len(nums1)):
    res.append(nums1[pointer1])
    pointer1 += 1

while(pointer2 < len(nums2)):
    res.append(nums2[pointer2])
    pointer2 += 1

return res

# O(N) and O(1)





## check if Algorithm is happy number -- Leetcode to check input output
def getSquareSum(n):
    n = list(str(n))
    
    res = 0
    for number in n:
        res += int(number)**2
    
    return res

Data = set()
while(n != 1):
    if(n not in Data):
        Data.add(n)
    else:
        return False
    n = getSquareSum(n)

return True



## Daily Temperatures
# For example, given the list of temperatures T = [73, 74, 75, 71, 69, 72, 76, 73], your output should be [1, 1, 4, 2, 1, 1, 0, 0]. 
# O(N^2)
res = []
for i in range(0, len(T)-1):
    Flag = False
    for j in range(i+1, len(T)):
        if(T[j] > T[i]):
            res.append(j - i)
            Flag = True
            break
    if(not Flag):
        res.append(0)
res.append(0)
return res
# O(N) and O(N)
ans = [0]*len(T)
stack = []

for i in range(len(T)-1, -1, -1):
    while(stack and T[i] >= T[stack[-1]]):
        stack.pop()
    if(stack):
        ans[i] = stack[-1] - i
    stack.append(i)        

return ans

## Group anagrams

# O(NKlogK) and O(NK)
Data = {}
for i in range(0, len(strs)):
    temp = ''.join(list(sorted(strs[i])))
    if(temp in Data):
        Data[temp].append(strs[i])
    else:
        Data[temp] = [strs[i]]
return Data.values()

# O(NK) and O(NK)
res = {}
for i in range(0, len(strs)):
    
    count = [0]*26
    for char in strs[i]:
        count[(ord(char) - 97)] += 1
    if(tuple(count) in res):
        res[tuple(count)].append(strs[i])
    else:
        res[tuple(count)] = [strs[i]]

return res.values()

## Word Search - DFS
def searchWord(board, x, y, word, pointer):
    # Base Cases
    if(pointer >= len(word)):
        return True
    if(y < 0 or y >= len(board) or x < 0 or x >= len(board[0]) or word[pointer] != board[y][x]):
        return False
    
    value = board[y][x]
    board[y][x] = ""
    # 4 choices up, down, right, left
    status = searchWord(board, x + 1, y, word, pointer + 1) or searchWord(board, x - 1, y, word, pointer + 1) or searchWord(board, x, y+1, word, pointer + 1) or searchWord(board, x, y-1, word, pointer + 1)
    board[y][x] = value
    return status

for y in range(0, len(board)):
    for x in range(0, len(board[0])):
        if(board[y][x] == word[0] and searchWord(board, x, y, word, 0)):
            return True

return False

## Generate Paranthesis
# BackTracking - O(4^n/sqrt(n​) both time and space
res = []
def genParanthesis(sentence, left, right):
    if(len(sentence) == 2*n):
        res.append(sentence)
        return
    if(left < n):
        genParanthesis(sentence + '(', left + 1, right)
    if(right < left):
        genParanthesis(sentence + ')', left, right+1)

genParanthesis('', 0, 0)
return res

# Maximal Square - O(n^2)
Input: 
1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0
Output: 4
# DP
if(len(matrix) == 0):
    return 0
# result
maxSquare = 0
# Init and Base Case setting
DP = [[0 for x in range(0, len(matrix[0]))] for y in range(0, len(matrix))]
for y in range(0, len(matrix)):
    for x in range(0, len(matrix[y])):
        if(y == 0 or x == 0):
            DP[y][x] = int(matrix[y][x])
            if(maxSquare < DP[y][x]):
                maxSquare = DP[y][x]
# DP[y][x] = min(matrix[y-1][x-1], matrix[y][x-1], matrix[y-1][x]) if matrix[y][x] == 1
for y in range(1, len(matrix)):
    for x in range(1, len(matrix[y])):
        if(matrix[y][x] == '1'):
            DP[y][x] = int(min(min(DP[y-1][x], DP[y][x-1]), DP[y-1][x-1])) + 1
            if(maxSquare < DP[y][x]):
                maxSquare = DP[y][x]
return maxSquare**2

## How to - https://leetcode.com/problems/top-k-frequent-elements/
## https://leetcode.com/problems/decode-string/

## Container with most water
# O(n^2)
mostWater = 0
for i in range(0, len(height)-1):
    maxContainer = 0
    for j in range(i+1, len(height)):
        maxContainer = max(maxContainer, min(height[i], height[j]) * (j - i))
    mostWater = max(maxContainer, mostWater)
return mostWater
# Two pass - O(N)
left = 0
right = len(height) - 1

maxArea = 0
while(left < right):
    
    contWidth = right - left
    contHeight = min(height[right], height[left])
    
    maxArea = max(maxArea, contWidth*contHeight)
    
    if(height[left] < height[right]):
        left += 1
    else:
        right -= 1
return maxArea

# Find firsrt and last position of element in sorted array
# O(N) and O(1)
Flag = False
res = []

for i in range(0, len(nums)):
    if(nums[i] == target and not Flag):
        Flag = True
        res.append(i)
    
    if(nums[i] != target and Flag):
        res.append(i-1)
        break

if(len(res) == 0):
    return [-1, -1]
elif(len(res) == 1 and nums[-1] == nums[res[0]]):
    res.append(len(nums)-1)
elif(len(res) == 1):
    res = res + res

return res

# O(logn) and O(1)
# Base Cases
if(len(nums) == 0):
    return [-1, -1]
if(target < nums[0]):
    return [-1, -1]
if(target > nums[-1]):
    return [-1, -1]
# Simple BS
def BinarySearch(nums, low, high, leftIndex, rightIndex, target):
    
    if(low > high):
        return [leftIndex, rightIndex]
    
    mid = (low + high)//2
    
    if(nums[mid] == target):
        leftIndex = min(leftIndex, mid)
        rightIndex = max(rightIndex, mid)
        
        leftPart = BinarySearch(nums, low, mid - 1, leftIndex, rightIndex, target)
        rightPart = BinarySearch(nums, mid + 1, high, leftIndex, rightIndex, target)
        
        leftIndex = min(leftPart[0], rightPart[0])
        rightIndex = max(leftPart[1], rightPart[1])
        
        return [leftIndex, rightIndex]
    elif(nums[mid] > target):
        return BinarySearch(nums, low, mid - 1, leftIndex, rightIndex, target)
    else:
        return BinarySearch(nums, mid + 1, high, leftIndex, rightIndex, target)

result = BinarySearch(nums, 0, len(nums)-1, len(nums)-1, 0, target)
if(result[0] > result[1]):
    return [-1, -1]
return result
        



######## Dyamic Prog

# Max prod sub array
        # Dynamic Programming O(N)
        if(len(nums) == 0):
            return 1
        
        maxPos = nums[0]
        maxNeg = nums[0]
        overallMax = nums[0]
        
        for num in nums[1:]:
            arr = [num, maxPos*num, maxNeg*num]
            maxPos = max(arr)
            maxNeg = min(arr)
            overallMax = max([overallMax, maxPos, maxNeg])
        
        return overallMax
        
        
        # Brute Force O(n^2)
        if(len(nums) == 0):
            return 0
        else:
            overallMax = nums[0]
            for i in range(0, len(nums)):
                prod = nums[i]
                for j in range(i+1, len(nums)):
                    if(prod > overallMax):
                        overallMax = prod
                    prod = prod*nums[j]
                if(prod > overallMax):
                    overallMax = prod
            return overallMax
        
        # Brute force
        overallMax = -2**31 + 1
        for i in range(0, len(nums)):
            maxProd = -2**31 + 1
            for j in range(i+1, len(nums)+1):
                temp = 1
                for num in nums[i:j]:
                    temp = temp*num
                if(temp > maxProd):
                    maxProd = temp
            if(maxProd > overallMax):
                overallMax = maxProd
        return overallMax

# Search in a 2d matrix

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

# x^n
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
# Majority element occuring more than n//2 times
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

# Occuring more than n//3 times
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
        
# power of 2, check if a number is power of 2

        # Bit manipulation
        return n & (n - 1) == 0 if( n != 0) else False
        
        # Bits method
        if(n <= 0):
            return False
        if(n == 1):
            return True
        binary = bin(n)
        
        if(int(binary[3:]) == 0):
            return True
        else:
            return False
        
        
        # Log method -- Not always accurate as built in log functions are error prone
        import math
        if((int(math.log(n,2)) - math.log(n,2)) == 0):
            return True
        else:
            return False
            
        
        # Brute Force
        if(n <= 0):
            return False
        while(n >= 1):
            if(n == 1):
                return True
            elif(n%2 == 0):
                n = n//2
            else:
                return False
        return True