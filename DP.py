## Min operations to reach N from 0, only 2 operations allowed +1 or *2
    # Iterative
    
    count = 0
    while(n):
        if(n%2 == 0):
            n = n//2
            count += 1
        else:
            n -= 1
            count += 1
    return count
    
    # Memoized
    DP = [-1 for x in range(n)]
    
    def solve(current, target):
        
        if(current > target):
            return 2**31 - 1
        
        if(current == target):
            return 0
        
        if(DP[current] != -1):
            return DP[current]
        
        DP[current] = 1 + min(solve(current + 1, target), solve(current*2, target))
        return DP[current]
    
    return 1 + solve(1, n)
    
    
    # Recursive
    def solve(current, target):
        
        if(current > target):
            return 2**31 - 1
        
        if(current == target):
            return 0
        
        return 1 + min(solve(current + 1, target), solve(current*2, target))
    
    return 1 + solve(1, n)

## Max length Chain, Pairs of Data structure are given s.t. (a, b) b > a and chain can form if a2 > b1
    class Pair(object):
        def __init__(self, a, b):
            self.a = a
            self.b = b
    Parr.sort(key = lambda x : x.b) # sorted is important
    
    def solve(pairArr, pointer, lastPair):
        
        if(pointer >= len(pairArr)):
            return 0
        
        # 2 choice select this pair or not
        currPair = pairArr[pointer]
        
        if(currPair.a > lastPair.b):
            return max(1 + solve(pairArr, pointer + 1, currPair), solve(pairArr, pointer + 1, lastPair))
        else:
            return solve(pairArr, pointer + 1, lastPair)
    
    return solve(Parr, 0, Pair(-1, -1))

## Optimal Strategy for Game -- Of a length of array even in size select either first or last element, 2 players u go first everyone increase their chances of wining

    # Choice Diagram
    pt1 = 0
    pt2 = size - 1
    
    ## Recursive

    # Chance represent boolena value if true then we pick else other player
    def solve(Arr, pt1, pt2, Chance):
        # Base Case
        if(pt1 > pt2):
            return 0
        
        # choices
        if(Chance):
            # We pick -- We Earn
            return max(Arr[pt1] + solve(Arr, pt1 + 1, pt2, not Chance), Arr[pt2] + solve(Arr, pt1, pt2 - 1, not Chance))
        else:
            # Other pick -- We dont Earn anything i.e. we earn least as he earn max
            return min(solve(Arr, pt1 + 1, pt2, not Chance), solve(Arr, pt1, pt2 - 1, not Chance))
       
    print(solve(Arr, pt1, pt2, True))

    
    ## Memoized version
    
    # 3D - [-1, -1] represent [Our Chance, Opposition Chance]
    DP = [[[-1, -1] for x in range(0, size + 1)] for y in range(0, size + 1)]
    
    # Chance represent boolena value if true then we pick else other player
    def solve(Arr, pt1, pt2, Chance):
        # Base Case
        if(pt1 > pt2):
            return 0
        
        # choices
        if(Chance):
            if(DP[pt1][pt2][0] != -1):
                return DP[pt1][pt2][0]
            DP[pt1][pt2][0] = max(Arr[pt1] + solve(Arr, pt1 + 1, pt2, not Chance), Arr[pt2] + solve(Arr, pt1, pt2 - 1, not Chance))
            return DP[pt1][pt2][0]
        else:
            if(DP[pt1][pt2][1] != -1):
                return DP[pt1][pt2][1]
            DP[pt1][pt2][1] = min(solve(Arr, pt1 + 1, pt2, not Chance), solve(Arr, pt1, pt2 - 1, not Chance))
            return DP[pt1][pt2][1]
       
    print(solve(Arr, pt1, pt2, True))

## Fog Hopping - 3 chooces 1, 2, 3

    Target = int(input())

    # choice diagram - 1 or 2 or 3 plus
    def solve(curr, target):
        # Base Cases
        if(curr == target):
            return 0
        # Corner Cases
        if(target - curr == 1):
            return 1
        elif(target - curr == 2):
            return 2
        elif(target - curr == 3):
            return 4
        # Main Return -- Recursive Calls
        return solve(curr + 1, target) + solve(curr + 2, target) + solve(curr + 3, target)
        
    print(solve(0, Target))

    # Memoized Version
    DP = [[-1, -1, -1] for x in range(0, Target + 1)]
    
    # choice diagram - 1 or 2 or 3 plus
    def solve(curr, target):
        # Base Cases
        if(curr == target):
            return 0
        # Corner Cases
        if(target - curr == 1):
            return 1
        elif(target - curr == 2):
            return 2
        elif(target - curr == 3):
            return 4
        
        if(DP[curr][0] == -1):
            DP[curr][0] = solve(curr + 1, target)
        if(DP[curr][1] == -1):
            DP[curr][1] = solve(curr + 2, target)
        if(DP[curr][2] == -1):
            DP[curr][2] = solve(curr + 3, target)
        
        return DP[curr][0] + DP[curr][1] + DP[curr][2]
        
    print(solve(0, Target))

## Min number of Jumps -- Given array of location vs jumps max you can go, if zero u cant go further. Reach end of it

    size = int(input())
    Arr = list(map(int, input().strip().split()))
    
    
    # O(n) Solution -- Greedy Approach
    
    if(size <= 1):
        print(0)
    elif(Arr[0] == 0):
        print(-1)
    else:
        
        Flag = False
        stepNumber = Arr[0]
        maxReach = Arr[0]
        jumps = 1 # first jump
        
        for i in range(1, len(Arr)):
            
            if(i == len(Arr)-1):
                print(jumps)
                Flag = True
                break
            
            maxReach = max(maxReach, i + Arr[i])
            
            stepNumber -= 1
            
            if(stepNumber == 0):
                jumps += 1
                if(i >= maxReach):
                    print(-1)
                    Flag = True
                    break
                stepNumber = maxReach - i
        if(not Flag):
            print(-1)
            
    
    # More Optimized Memoized version
    DP = [-1 for x in range(size + 1)]
    
    # Recursive
    def solve(pointer, Arr):
        # Base Case
        if(pointer == len(Arr) - 1):
            return 0
        # Corner Case
        if(pointer >= len(Arr)):
            return 2**31 - 1
        if(Arr[pointer] == 0):
            return 2**31 - 1
        
        # Memoized
        if(DP[pointer] != -1):
            return DP[pointer]
        
        minAns = 2**31 - 1
        for jump in range(1, Arr[pointer] + 1):
            if(pointer + jump < len(Arr)):
                if(DP[pointer + jump] == -1):
                    DP[pointer + jump] = 1 + solve(pointer + jump, Arr)
                minAns = min(minAns, DP[pointer + jump])
        
        DP[pointer] = minAns
        return DP[pointer]
    
    
    # Memoized
    DP = [-1 for x in range(size + 1)]
    
    # Recursive
    def solve(pointer, Arr):
        # Base Case
        if(pointer == len(Arr) - 1):
            return 0
        # Corner Case
        if(pointer >= len(Arr)):
            return 2**31 - 1
        if(Arr[pointer] == 0):
            return 2**31 - 1
        
        # Memoized
        if(DP[pointer] != -1):
            return DP[pointer]
        
        minAns = 2**31 - 1
        for jump in range(1, Arr[pointer] + 1):
            tempAns = 1 + solve(pointer + jump, Arr)
            minAns = min(minAns, tempAns)
        DP[pointer] = minAns
        return DP[pointer]
    
    
    # Recursive
    def solve(pointer, Arr):
        # Base Case
        if(pointer == len(Arr) - 1):
            return 0
        # Corner Case
        if(pointer >= len(Arr)):
            return 2**31 - 1
        if(Arr[pointer] == 0):
            return 2**31 - 1
        
        minAns = 2**31 - 1
        for jump in range(1, Arr[pointer] + 1):
            tempAns = 1 + solve(pointer + jump, Arr)
            minAns = min(minAns, tempAns)
        return minAns
    
    res = solve(0, Arr)
    
    if(res > len(Arr)):
        print(-1)
    else:
        print(res)
     
## Longest Increasing Subsequence

    Size = int(input())
    Arr = list(map(int, input().strip().split()))
    
    # Memoized
    DP = [{} for x in range(Size + 1)] # this Datastructure is important
    
    def solve(Arr, pointer, parent):
        
        if(pointer >= len(Arr)):
            return 0
        if(parent in DP[pointer]):
            return DP[pointer][parent]
        
        if(parent >= Arr[pointer]):
            DP[pointer][parent] = solve(Arr, pointer + 1, parent)
            return DP[pointer][parent]
        
        DP[pointer][parent] = max(1 + solve(Arr, pointer + 1, Arr[pointer]), solve(Arr, pointer + 1, parent))
        return DP[pointer][parent]
    
    print(solve(Arr, 0, -1))
    
    
    
    # Recursive
    def solve(Arr, pointer, parent):
        
        if(pointer >= len(Arr)):
            return 0
        
        if(parent >= Arr[pointer]):
            return solve(Arr, pointer + 1, parent)
        
        return max(1 + solve(Arr, pointer + 1, Arr[pointer]), solve(Arr, pointer + 1, parent))
    
    print(solve(Arr, 0, -1))

Minimum number of Coins
Minimum sum partition 