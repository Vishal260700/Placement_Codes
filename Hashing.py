## Uncommon charecters in 2 string and return in sorted order
Dict1 = [0 for x in range(26)]
Dict2 = [0 for x in range(26)]

for char in A:
    Dict1[ord(char) - 97] += 1

for char in B:
    Dict2[ord(char) - 97] += 1

res = ""
for i in range(0, 26):
    if(Dict1[i] != Dict2[i]):
        if(Dict1[i] == 0 or Dict2[i] == 0):
            res += chr(i + 97)

return res if (len(res)) else -1

## Zero Sum SubArrays - Get all subarrays in an array which sum to zero

## Thought process betwen i and j sum of elements is zero means cumm sum till 'i' == cum sum till 'j'

testCases = int(input())

while(testCases > 0):
    
    size = int(input())
    arr = list(map(int, input().strip().split()))
    
    count = 0
    total = 0
    
    
    Dict = {0 : 1}
    
    total = 0
    count = 0
    
    for i in range(size):
        
        total += arr[i]
        
        if(total in Dict):
            count += Dict[total]
            Dict[total] += 1
        else:
            Dict[total] = 1
    
    print(count)
    
    testCases -= 1

## Longest Consecutive SubSequence
Input:
2
7
2 6 1 9 4 5 3
7
1 9 3 10 4 20 2

Output:
6
4
# O(nlogn)
arr.sort()
    
maxLen = 0
tempMax = 0
for i in range(1, size):
    if(arr[i] == arr[i-1]):
        continue
    
    if (abs(arr[i] - arr[i-1]) == 1):
        tempMax += 1
    else:
        maxLen = max(maxLen, tempMax)
        tempMax = 0
print(max(maxLen, tempMax) + 1)

# O(n) and O(n)
arrSet = set(arr)
maxLen = 0
for i in range(size):
    
    if(arr[i] - 1 not in arrSet):
        
        # j -> new start
        j = arr[i]
        while(j in arrSet):
            j += 1
        
        maxLen = max(maxLen, j - arr[i])
    
print(maxLen)

## Divide arrays in pairs (2) such that their sum is divisible by K (all of them)
Input:
2
4
9 7 5 3
6
4
91 74 66 48
10

Output:
True
False
if(size % 2 != 0):
        print('False')
else:
    
    elementSet = {}
    for i in range(size//2):
        if(arr[i]%K not in elementSet):
            elementSet[arr[i]%K] = 1
        else:
            elementSet[arr[i]%K] += 1
    
    Flag = False
    
    for i in range(size//2, size):
        
        if(arr[i]%K == 0 and arr[i]%K in elementSet):
            if(elementSet[0] > 0):
                elementSet[0] -= 1
            else:
                Flag = True
                print('False')
                break
        elif(K - arr[i]%K in elementSet):
            if(elementSet[K - arr[i]%K] > 0):
                elementSet[K - arr[i]%K] -= 1
            else:
                Flag = True
                print('False')
                break
        else:
            Flag = True
            print('False')
            break
    
    if(not Flag):
        print('True')

## Four Sum problem

# Recursive

testCases = int(input())

while(testCases > 0):
    
    Queries = list(map(int, input().strip().split()))
    Arr = list(map(int, input().strip().split()))
    K = Queries[1]
    
    # 4 elements whose sum is equal to K
    
    # self.res = []
    
    def solve(capacity, total, Arr, pointer, K, tempRes):
        
        if(capacity == 0 and total == K):
            print(tempRes, end = "$")
            return
            
        if(pointer >= len(Arr)):
            return 
        
        # choice to select this elemnt or not
        solve(capacity - 1, total + Arr[pointer], Arr, pointer + 1, K, tempRes + str(Arr[pointer]))
        solve(capacity, total, Arr, pointer + 1, K, tempRes)
        
        return 
    
    solve(4, 0, Arr, 0, K, "")
    
    testCases -= 1

# Hashing











## Largets SubArray with sum = 0

## Brute Force
length = -1
for i in range(0, len(arr)-1):
    for j in range(i+1, len(arr)):
        if(sum(arr[i:j]) == 0):
            if(length < len(arr[i:j])):
                length = len(arr[i:j])
return length

# O(N) and O(N)
Dict = {}
total = 0
maxLen = 0
Dict[0] = -1 # takes into account if the sum of array from begining to 'i' is zero
for i in range(0, n):
    total += arr[i]
    if (total in Dict):
        tempLen = i - Dict[total]
        maxLen = max(maxLen, tempLen)
    else:
        Dict[total] = i # first occurence of cummSum
return maxLen

## Common elements in 3 arrays (sorted)
     
# Hashing - O(N) and O(N)
Dict = {} # [a, b, c]
for i in range(0, n1):
    if(A[i] not in Dict):
        Dict[A[i]] = [i]

for i in range(0, n2):
    if(B[i] in Dict and len(Dict[B[i]]) == 1):
        Dict[B[i]].append(i)

res = []
for i in range(0, n3):
    if(C[i] in Dict and len(Dict[C[i]]) == 2):
        Dict[C[i]].append(i)
        res.append(C[i])

return res

# O(N) and O(1)
a = 0
b = 0
c = 0
res = []
while(a < n1 and b < n2 and c < n3):
    if(A[a] == B[b] and B[b] == C[c]):
        res.append(A[a])
        a += 1
        b += 1
        c += 1
    elif(A[a] < B[b]):
        a += 1
    elif(B[b] < C[c]):
        b += 1
    else:
        c += 1
return res

# sort of elements of array by frequency

testCases = int(input())

while(testCases > 0):
    
    size = int(input())
    Arr = list(map(int, input().strip().split()))
    
    FreqTable = [[] for x in range(0, len(Arr))]
    
    Dict = {}
    for element in Arr:
        if(element not in Dict):
            Dict[element] = 1
        else:
            Dict[element] += 1
    
    for key in Dict.keys():
        FreqTable[Dict[key]].append(key)
    
    for i in range(len(FreqTable)-1, -1, -1):
        if(len(FreqTable[i])):
            for e in FreqTable[i]:
                times = i            
                while(times):
                    print(e, end = " ")
                    times -= 1
    
    print('')
    
    testCases -= 1
## Relative Sorting

# given 2 arrays sort based on ordering of array2 and add leftover of array1 to end of result in sorted manner
Input:
2
11 4
2 1 2 5 7 1 9 3 6 8 8
2 1 8 3
8 4
2 6 7 5 2 6 8 4 
2 6 4 5
Output:
2 2 1 1 8 8 3 5 6 7 9
2 2 6 6 4 5 7 8
testCases = int(input())

while(testCases > 0):
    
    Queries = list(map(int, input().strip().split()))
    
    SuperArr = list(map(int, input().strip().split()))
    SubArr = list(map(int, input().strip().split()))
    
    Dict = {}
    for element in SuperArr:
        if(element not in Dict):
            Dict[element] = 1
        else:
            Dict[element] += 1
    
    for element in SubArr:
        if(element in Dict):
            times = Dict[element]
            while(times > 0):
                print(element, end=" ")
                times -= 1
            del Dict[element]
    
    leftOver = []
    for key in Dict.keys():
        times = Dict[key]
        while(times > 0):
            leftOver.append(key)
            times -= 1
    
    leftOver.sort()
    for element in leftOver:
        print(element, end = " ")
    
    print('')
    
    testCases -= 1

# Min loss
# given an array of stock prices, buy and sell such that the loss is minimum no profit
    
Input - 
    5
    20 7 8 2 5

Output - 2 (7 - 5)

    # O(nlogn) - sort and hashmap

    Dict = {}
    for i in range(0, len(price)):
        if(price[i] not in Dict):
            Dict[price[i]] = i
    
    price = list(sorted(price, reverse = True))
    minLoss = 2**31 - 1
    for i in range(0, len(price) - 1):
        if(Dict[price[i]] < Dict[price[i+1]]):
            minLoss = min(minLoss, abs(price[i] - price[i+1]))
    
    return minLoss


    # O(n^2)
    minLoss = 2**31 - 1

    for i in range(0, len(price)-1):
        for j in range(i + 1, len(price)):
            if(price[j] <= price[i]):
                tempMax = price[i] - price[j]
                minLoss = min(tempMax, minLoss)
    
    return minLoss

https://practice.geeksforgeeks.org/problems/check-frequencies4211/1
https://practice.geeksforgeeks.org/problems/smallest-window-in-a-string-containing-all-the-characters-of-another-string/0
https://practice.geeksforgeeks.org/problems/swapping-pairs-make-sum-equal4142/1
https://practice.geeksforgeeks.org/problems/find-all-four-sum-numbers/0
https://practice.geeksforgeeks.org/problems/check-frequencies4211/1