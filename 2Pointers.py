## remove Duplicates from sorted Array

    i = 0
    while(i < len(nums)-1):
        if(nums[i] == nums[i+1]):
            nums.pop(i)
        else:
            i += 1
    
    return i+1

## Max consecurtive 1s

    # Brute Force - O(N) O(1)
    overallMax = 0
    tempMax = 0
    
    for num in nums:
        if(num == 1):
            tempMax += 1
        else:
            overallMax = max(overallMax, tempMax)
            tempMax = 0
    
    return max(overallMax, tempMax)

## Trapping Rain Water -- V.Imp

        # Brute force -- concept for any i in [1, len-1] uske leftka max and rightka max buildings mein jo chota hai utna paani bharega uske upar i.e. ->
        water = 0
        for i in range(1, len(height)-1):
            lmax = max(height[:i])
            rmax = max(height[i+1:])
            temp = min(lmax, rmax) - height[i]
            if(temp > 0):
                water = water + temp
        return water

        # Optimized - O(N), O(N) - DP
        # storing lmax and rmax -- 3 pass algo
        lmax = [0 for x in range(0, len(height))]
        rmax = [0 for x in range(0, len(height))]
        
        for i in range(1, len(height)):
            lmax[i] = max(height[i-1], lmax[i-1])
        
        for i in range(len(height)-2, -1, -1):
            rmax[i] = max(height[i+1], rmax[i+1])
        
        storage = 0
        for i in range(0, len(height)):
            water = min(lmax[i], rmax[i]) - height[i]
            if(water > 0):
                storage += water
        return storage

        ## Two Pointers -- check later


## Sort colors - https://leetcode.com/problems/sort-colors/discuss/698049/Easy-Python3-Solution-Using-3-pointer-(-Dutch-national-flag-Algo)-or-One-pass-Algorithm

# O(n) - Dutch Flag Algorithm
zeroPointer = 0
onePointer = 0
twoPointer = len(nums) - 1

while(onePointer <= twoPointer):
    if(nums[onePointer] == 0):
        nums[zeroPointer], nums[onePointer] = nums[onePointer], nums[zeroPointer]
        zeroPointer += 1
        onePointer += 1
    elif(nums[onePointer] == 1):
        onePointer += 1
    else:
        nums[onePointer], nums[twoPointer] = nums[twoPointer], nums[onePointer]
        twoPointer -= 1

# O(n) and O(n)
zeroes = 0
ones = 0
twos = 0
for num in nums:
    if(num == 0):
        zeroes += 1
    elif(num == 1):
        ones += 1
    else:
        twos += 1

res1 = [0 for x in range(zeroes)]
res2 = [1 for x in range(ones)]
res3 = [2 for x in range(twos)]
print(res1 + res2 + res3)


# simple sorting  - O(nlogn)
nums.sort()


## Sum of Perfect Squares
# Recursove
def isSquare(n):
    if(int(n) - n == 0):
        return True
    return False

def sumPrimeSquares(n, count):
    if(n == 0):
        return 0
    
    if(isSquare(math.sqrt(n))):
        return count + 1
    
    res = []
    for i in range(1, n//2 + 1):
        res.append(sumPrimeSquares(n-i, count) + sumPrimeSquares(i, count))
    return min(res)
        
return sumPrimeSquares(n, 0)
# DP - O(n^2) and O(n)
DP = [-1 for x in range(0, n+1)]
DP[0] = 0
DP[1] = 1

for i in range(2, n+1):
    if((math.sqrt(i)*10)%10 == 0):
        DP[i] = 1
    else:
        temp = []
        for j in range(1, i//2 + 1):
            temp.append(DP[i-j] + DP[j])
        DP[i] = min(temp)

return DP[-1]

# Better DP
dp = [float('inf')] * (n+1) # max value
dp[0] = 0 # Base Case
square_nums = [i**2 for i in range(0, int(math.sqrt(n))+1)] # All squares till given value

for i in range(1, n+1):
    for square in square_nums:
        if(i < square): break # doesn't matter
        dp[i] = min(dp[i], dp[i-square] + 1) # +1 is for that square we are substracting.
return dp[-1]

# Find All anagrams in a string
# Brute Force - O(N^2) and O(N)
# Corner Case
if(len(s) < len(p)):
    return []

res = []
for i in range(0, len(s) - len(p) + 1):
    # to check
    tempArr = s[i:i + len(p)]
    # Dict of check String
    checkDict = {}
    for char in p:
        if(char in checkDict):
            checkDict[char] += 1
        else:
            checkDict[char] = 1
    Flag = True
    for char in tempArr:
        if(char in checkDict):
            if(checkDict[char] <= 0):
                Flag = False
                break
            else:
                checkDict[char] -= 1
        else:
            Flag = False
            break
    if(Flag):
        if(sum(checkDict.values()) == 0):
            res.append(i)
    
return res
# Count Sort
# Map of Charecters
truth = [0 for x in range(26)]
test = [0 for x in range(26)]

# Fill the real truth we need to check
for char in p:
    truth[ord(char) - ord('a')] += 1

# Fill up elements as per map scenario
res = []
for i in range(0, len(s) - len(p) + 1):
    if(i > 0):
        test[ord(s[i-1]) - ord('a')] -= 1
        test[ord(s[i + len(p) - 1]) - ord('a')] += 1
    else:
        # first time ever
        for j in range(0, len(p)):
            test[ord(s[j]) - ord('a')] += 1
    
    if(test == truth):
        res.append(i)

return res

## Subarray Sum Equals K
# Best O(N) and O(N)
Dict = {}
res = 0
total = 0
for i in range(0, len(nums)):
    total += nums[i]
    if(total == k):
        res += 1
    if(total - k in Dict):
        res += Dict[total - k] + 1
    if(total in Dict):
        Dict[total] += 1
    else:
        Dict[total] = 0
return res

# Optimized O(N^2)
count = 0
for i in range(0, len(nums)):
    total = 0
    for j in range(i, len(nums)):
        total += nums[j]
        if(total == k):
            count += 1
return count


# Brute force O(N^3)
window = 1
count = 0
while(window <= len(nums)):         
    for i in range(0, len(nums) - window + 1):
        sumSubArr = sum(nums[i: i + window])
        if(sumSubArr == k):
            count += 1
    window += 1
return count







