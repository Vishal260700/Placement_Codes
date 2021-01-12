#### KnapSack - Fractional, 0-1, Unbounded

    ### o-1 Knapsack

        # Recursive O(2^n), O(1)
        def knapsack(values, weights, capacity, pointer):
            
            if(pointer < 0 or capacity == 0):
                return 0
            
            if(weights[pointer] > capacity):
                return knapsack(values, weights, capacity, pointer - 1)
            
            return max(values[pointer] + knapsack(values, weights, capacity - weights[pointer], pointer - 1), knapsack(values, weights, capacity, pointer - 1))

        print(knapsack(values, weights, capacity, items - 1))

        # Memoization O(n*w), O(n*w)
        dp = [[None for x in range(0, capacity + 1)] for y in range(0, items + 1)]
            
        def knapsack(values, weights, capacity, pointer):
            
            if(pointer < 0 or capacity == 0):
                return 0
            
            if(weights[pointer] > capacity):
                if(dp[pointer][capacity] is None):
                    dp[pointer][capacity] = knapsack(values, weights, capacity, pointer - 1)
                return dp[pointer][capacity]
            if(dp[pointer][capacity] is None):
                dp[pointer][capacity] = max(values[pointer] + knapsack(values, weights, capacity - weights[pointer], pointer - 1), knapsack(values, weights, capacity, pointer - 1))
            return dp[pointer][capacity]

        print(knapsack(values, weights, capacity, items - 1))

        # Tabulation -- Takes care of recursive limit reached error

        dp = [[None for x in range(0, capacity + 1)] for y in range(0, items + 1)] # items = len(values array)
        
        for i in range(0, items+1):
            for j in range(0, capacity+1):
                if(i == 0 or j == 0): # base case
                    dp[i][j] = 0
                else:
                    if(weights[i-1] > j):
                        dp[i][j] = dp[i-1][j]
                    else:
                        dp[i][j] = max(values[i-1] + dp[i-1][j-weights[i-1]], dp[i-1][j])
        
        print(dp[items][capacity])

    ## Subset Sum - given targetSum find if there is any such subset

        # Recursion
        def subsetSum(Arr, total, pointer, target):
                
            if(total == target):
                return True
            if(pointer >= len(Arr)):
                return False
            
            if(Arr[pointer] > target):
                return subsetSum(Arr, total, pointer + 1, target)
            
            return subsetSum(Arr, total + Arr[pointer], pointer + 1, target) or subsetSum(Arr, total, pointer + 1, target)
        
        return (1 if(subsetSum(A, 0, 0, B)) else 0)

        # DP - Tabulation
        dp = [[None for x in range(0, B + 1)] for y in range(0, len(A) + 1)]
            
            for x in range(0, B + 1):
                dp[0][x] = False
            
            for y in range(0, len(A) + 1):
                dp[y][0] = True
            
            for y in range(1, len(A) + 1):
                for x in range(1, B + 1):
                    if(A[y-1] > x):
                        dp[y][x] = dp[y-1][x]
                    else:
                        dp[y][x] = dp[y-1][x] or dp[y-1][x - A[y-1]]
            
            return (1 if(dp[len(A)][B]) else 0)

    ## Equal Sum Partition

        # Recursion
        arr = list(map(int, input().strip().split()))
            
        def subsetSum(arr, total1, total2, pointer):
            if(pointer >= len(arr) and total1 == total2):
                return True
            elif(pointer >= len(arr) and total1 != total2):
                return False
            
            return (subsetSum(arr, total1 + arr[pointer], total2, pointer + 1) or subsetSum(arr, total1, total2 + arr[pointer], pointer + 1))
            
        if(subsetSum(arr, 0, 0, 0)):
            print('YES')
        else:
            print('NO')
        
        # DP - Tabulation
        arr = list(map(int, input().strip().split()))
        
        if(sum(arr)%2 != 0):
            print('NO')
        else:
            target = int(sum(arr)/2)
            
            # same problem of subsetsum
            dp = [[None for x in range(0, target+1)] for y in range(0, len(arr) + 1)]
            
            for x in range(0, target + 1):
                dp[0][x] = False
            
            for y in range(0, len(arr) + 1):
                dp[y][0] = True
            
            for y in range(1, len(arr) + 1):
                for x in range(1, target + 1):
                    if(arr[y-1] > x):
                        dp[y][x] = dp[y-1][x]
                    else:
                        dp[y][x] = dp[y-1][x] or dp[y-1][x - arr[y-1]]
            
            if(dp[len(arr)][target]):
                print('YES')  
            else:
                print('NO')

    ## Count of subset sum

        # DP
        arr = list(map(int, input().strip().split()))
        target = int(input())
        
        dp = [[0 for x in range(0, target + 1)] for y in range(0, len(arr) + 1)]
        
        for x in range(0, target + 1):
            dp[0][x] = 0
        
        for y in range(0, len(arr) + 1):
            dp[y][0] = 1
        
        for y in range(1, len(arr) + 1):
            for x in range(1, target + 1):
                if(arr[y - 1] > x):
                    dp[y][x] = dp[y-1][x]
                else:
                    dp[y][x] = dp[y-1][x] + dp[y-1][x - arr[y-1]]
        
        print(dp[len(arr)][target])

    # Both same (x + y = sum and x - y = given)
    ## Target Sum
    ## number of subset and given difference
        
        # DP
        dp = [[0 for x in range(0, S+1)] for y in range(0, len(nums) + 1)]
        
        target = (sum(nums) + S)
        
        if(target % 2 != 0):
            return 0
        else:
            
            target = int(target/2)
            
            dp = [[0 for x in range(0, target + 1)] for y in range(0, len(nums) + 1)]
            
            for x in range(0, target + 1):
                dp[0][x] = 0
            
            for y in range(0, len(nums) + 1):
                dp[y][0] = 1
            
            for y in range(1, len(nums) + 1):
                for x in range(1, target + 1):
                    if(nums[y-1] > x):
                        dp[y][x] = dp[y-1][x]
                    else:
                        dp[y][x] = dp[y-1][x] + dp[y-1][x - nums[y-1]]
            
            return dp[len(nums)][target]

    ## Minimum subset sum -- subset with min sum difference

#### Unbounded Knapsack

#### LCS

    # In subsequence order matter but continuity doesn't matter
    # In substring both matters

    ## Recursive
        def LCS(Arr1, Arr2, pointer1, pointer2):
            # Base Case
            if(pointer1 >= len(Arr1) or pointer2 >= len(Arr2)):
                return 0
            
            # Choice diagram
            if(Arr1[pointer1] == Arr2[pointer2]):
                return 1 + LCS(Arr1, Arr2, pointer1 + 1, pointer2 + 1)
            else:
                return max(LCS(Arr1, Arr2, pointer1 + 1, pointer2), LCS(Arr1, Arr2, pointer1, pointer2 + 1))
        
        print(LCS(arr1, arr2, 0, 0))

    ## DP

        dp = [[-1 for x in range(0, len(arr1)+1)] for y in range(0, len(arr2)+1)]
        
        # Base condition
        for x in range(0, len(arr1)+1):
            dp[0][x] = 0
        
        for y in range(0, len(arr2)+1):
            dp[y][0] = 0
        
        for y in range(1, len(arr2)+1):
            for x in range(1, len(arr1)+1):
                if(arr1[x-1] == arr2[y-1]):
                    dp[y][x] = 1 + dp[y-1][x-1]
                else:
                    dp[y][x] = max(dp[y-1][x], dp[y][x-1])
        
        print(dp[len(arr2)][len(arr1)])

    ### Longest common substring - continous in nature variation

        dp = [[-1 for x in range(0, len(arr1) + 1)] for y in range(0, len(arr2) + 1)]
        
        # Main DP code
        for y in range(0, len(arr2) + 1):
            for x in range(0, len(arr1) + 1):
                if(x == 0 or y == 0):
                    dp[y][x] = 0 ## Only change
                elif(arr1[x-1] == arr2[y-1]):
                    dp[y][x] = 1 + dp[y-1][x-1]
                else:
                    dp[y][x] = 0
        
        # Get max
        maxElement = -1
        for y in range(len(arr2)+1):
            temp = max(dp[y])
            if(maxElement < temp):
                maxElement = temp
        
        print(maxElement)

    ## Printing Longest common subsequence -- V.imp
        ## DP
        ## Same LCS
        dp = [[-1 for x in range(0, len(a) + 1)] for y in range(0, len(b)+1)]

        for y in range(0, len(b) + 1):
            for x in range(0, len(a) + 1):
                if(x == 0 or y == 0):
                    dp[y][x] = 0
                elif(a[x-1] == b[y-1]):
                    dp[y][x] = dp[y-1][x-1] + 1
                else:
                    dp[y][x] = max(dp[y-1][x], dp[y][x-1])
        ## This part is important
        x = len(a)
        y = len(b)
        res = ""
        while(x > 0 and y > 0):
            if(a[x-1] == b[y-1]):
                res = str(a[x-1]) + res
                x -= 1
                y -= 1
            else:
                if(dp[y-1][x] > dp[y][x-1]):
                    y -= 1
                    x = x
                else:
                    y = y
                    x -= 1
        return (res)

    # Shortest common supersequence - LCS(str1, str2) + (str1 - LCS(str1, str2)) + (str2 - LCS(str1, str2)) - Good Thought
    # Input:   str1 = "geek",  str2 = "eke"
    # Output: "geeke" (5 length)
        # DP
        arr1 = list(str1)
        arr2 = list(str2)
        
        dp = [[-1 for x in range(0, len(arr1) + 1)] for y in range(0, len(arr2) + 1)]

        for y in range(0, len(arr2) + 1):
            for x in range(0, len(arr1) + 1):
                if(x == 0 or y == 0):
                    dp[y][x] = 0
                elif(arr1[x-1] == arr2[y-1]):
                    dp[y][x] = 1 + dp[y-1][x-1]
                else:
                    dp[y][x] = max(dp[y-1][x], dp[y][x-1])

        return (len(arr1) + len(arr2) - dp[len(arr2)][len(arr1)])

    ## Printing the above -- VVIMP
            # Get LCS
            arr1 = list(str1)
            arr2 = list(str2)
            
            dp = [[-1 for x in range(0, len(arr1) + 1)] for y in range(0, len(arr2) + 1)]
        
            for y in range(0, len(arr2) + 1):
                for x in range(0, len(arr1) + 1):
                    if(x == 0 or y == 0):
                        dp[y][x] = 0
                    elif(arr1[x-1] == arr2[y-1]):
                        dp[y][x] = 1 + dp[y-1][x-1]
                    else:
                        dp[y][x] = max(dp[y-1][x], dp[y][x-1])
            # Get LCS Print -- with modifications
            x = len(arr1)
            y = len(arr2)
            res = ""
            
            while(x > 0 and y > 0):
                if(arr1[x-1] == arr2[y-1]):
                    res = str(arr1[x-1]) + res
                    x -= 1
                    y -= 1
                else:
                    if(dp[y-1][x] > dp[y][x-1]):
                        res = str(arr2[y-1]) + res # add less wala
                        y -= 1
                    else:
                        res = str(arr1[x-1]) + res # add less wala
                        x -= 1
            # if any left
            while(x > 0):
                res = str(arr1[x-1]) + res
                x -= 1
            # if any left
            while(y > 0):
                res = str(arr2[y-1]) + res
                y -= 1
            
            return res

    ## Insertion deletion allowed - Minimum Number of Insertion and Deletion to convert String a to String b
    # Input : str1 = "geeksforgeeks", str2 = "geeks"
    # Output : Minimum Deletion = 8, Minimum Insertion = 0  
        
        # DP
        dp = [[-1 for x in range(0, len(str1) + 1)] for y in range(0, len(str2) + 1)]
        
        for y in range(0, len(str2) + 1):
            for x in range(0, len(str1) + 1):
                if (x == 0 or y == 0):
                    dp[y][x] = 0
                elif(str2[y-1] == str1[x-1]):
                    dp[y][x] = 1 + dp[y-1][x-1]
                else:
                    dp[y][x] = max(dp[y-1][x], dp[y][x-1])
        
        print(len(str1) + len(str2) - 2*dp[len(str2)][len(str1)])

    ## Longest Palindromic Sequence
        # DP
        # Below 2 lines are the important part
        arr1 = string
        arr2 = list(reversed(string))
            
        dp = [[-1 for x in range(0, len(arr1) + 1)] for y in range(0, len(arr2) + 1)]
        
        for y in range(0, len(arr2) + 1):
            for x in range(0, len(arr1) + 1):
                if(x == 0 or y == 0):
                    dp[y][x] = 0
                elif(arr1[x-1] == arr2[y-1]):
                    dp[y][x] = 1 + dp[y-1][x-1]
                else:
                    dp[y][x] = max(dp[y-1][x], dp[y][x-1])
        
        return dp[len(arr2)][len(arr1)]

    ## Min number of deletions to make a string palindromic
        
        # use origial test string and its reverse part
        # Same LCM bas different return statement
        dp = [[0 for x in range(0, len(test) + 1)] for y in range(0, len(revTest) + 1)]

        for y in range(0, len(revTest) + 1):
            for x in range(0, len(test) + 1):
                if(x == 0 or y == 0):
                    dp[y][x] = 0
                elif(test[x-1] == revTest[y-1]):
                    dp[y][x] = 1 + dp[y-1][x-1]
                else:
                    dp[y][x] = max(dp[y-1][x], dp[y][x-1])

        return (len(test) - dp[len(revTest)][len(test)]) ## Logic

    ## Min number of insertions to make a string palindromic
    # Same code as min number of deletions but identifying and thinking of pairing in this where as in deletions removal of odd pairs is done

    # Longest Repeating Sub-Sequence

        # A -> given string
        str1 = list(A)
        str2 = str1
        
        dp = [[-1 for x in range(0, len(str1) + 1)] for y in range(0, len(str2) + 1)]
        
        for y in range(0, len(str2) + 1):
            for x in range(0, len(str1) + 1):
                if(x == 0 or y == 0):
                    dp[y][x] = 0
                elif((str1[x-1] == str2[y-1]) and (x != y)): # Only this change -- imp idea
                    dp[y][x] = 1 + dp[y-1][x-1]
                else:
                    dp[y][x] = max(dp[y-1][x], dp[y][x-1])
        
        return dp[len(str2)][len(str1)]

    # check if a string s is subseq of t -- Pattern matching
            # DP - Better with simple Two pointer approach
            # LCS of s and t is same as s then okay else false
            dp = [[-1 for x in range(0, len(s) + 1)] for y in range(0, len(t) + 1)]
            
            for y in range(0, len(t) + 1):
                for x in range(0, len(s) + 1):
                    if(x == 0 or y == 0):
                        dp[y][x] = 0
                    elif(t[y-1] == s[x-1]):
                        dp[y][x] = 1 + dp[y-1][x-1]
                    else:
                        dp[y][x] = max(dp[y-1][x], dp[y][x-1])
            
            if(dp[len(t)][len(s)] == len(s)):
                return True
            else:
                return False
            

    ## Edit Distance

#### Matrix Chain multiplication

    # Recursive

    def solve(Arr, leftIndex, rightIndex):
        # base case
        if(leftIndex >= rightIndex):
            return 0

        minAns = 2**31 - 1
        for k in range(leftIndex, rightIndex):
            # Dividing recursively
            tempAns = solve(Arr, leftIndex, k) + solve(Arr, k+1, rightIndex) + Arr[leftIndex-1]*Arr[k]*Arr[rightIndex] # last value is the main cost function
            minAns = min(minAns, tempAns)
        
        return minAns

    return solve(Arr, 1, len(Arr)-1)

    ## Memoization -- 4 line change in recursive

    DP = [[-1 for x in range(0, len(Arr) + 1)] for y in range(0, len(Arr) + 1)]

    def solve(Arr, leftIndex, rightIndex):
        # base case
        if(leftIndex >= rightIndex):
            return 0
        
        # New line
        if(DP[leftIndex][rightIndex] != -1):
            return DP[leftIndex][rightIndex]

        minAns = 2**31 - 1
        for k in range(leftIndex, rightIndex):
            # Dividing recursively
            tempAns = solve(Arr, leftIndex, k) + solve(Arr, k+1, rightIndex) + Arr[leftIndex-1]*Arr[k]*Arr[rightIndex] # last value is the main cost function
            minAns = min(minAns, tempAns)
        # New line
        DP[leftIndex][rightIndex]  = minAns
        # New return
        return DP[leftIndex][rightIndex] 

    return solve(Arr, 1, len(Arr)-1)

    ## Palindromic partitioning

    # helper
    def isPalindrome(string, l, r):
        while(l < r):
            if(string[l] != string[r]):
                return False
            l += 1
            r -= 1
        return True

    # min partitioning
    def solve(string, leftIndex, rightIndex):
        
        if(leftIndex >= rightIndex):
            return 0 # already palindrome
        
        if(isPalindrome(string, leftIndex, rightIndex)):
            return 0
        
        minAns = 2**31 - 1
        for k in range(leftIndex, rightIndex):
            tempAns = solve(string, leftIndex, k) + solve(string, k+1, rightIndex) + 1
            minAns = min(minAns, tempAns)
        
        return minAns

    return solve(str, 0, len(str)-1)

    ## Memoized version
    DP = [[-1 for x in range(0, len(str) + 1)] for y in range(0, len(str) + 1)]

    def isPalindrome(string, l, r):
        while(l < r):
            if(string[l] != string[r]):
                return False
            l += 1
            r -= 1
        return True

    # min partitioning
    def solve(string, leftIndex, rightIndex):
        
        if(leftIndex >= rightIndex):
            return 0 # already palindrome
        
        # New line
        if(DP[leftIndex][rightIndex] != -1):
            return DP[leftIndex][rightIndex]
        
        if(isPalindrome(string, leftIndex, rightIndex)):
            DP[leftIndex][rightIndex] = 0 # New Line
            return 0
        
        minAns = 2**31 - 1
        for k in range(leftIndex, rightIndex):
            tempAns = solve(string, leftIndex, k) + solve(string, k+1, rightIndex) + 1
            minAns = min(minAns, tempAns)
        
        # New line
        DP[leftIndex][rightIndex] = minAns
        return DP[leftIndex][rightIndex]

    return solve(str, 0, len(str)-1)

    DP = [[-1 for x in range(0, len(str) + 1)] for y in range(0, len(str) + 1)]
            
            def isPalindrome(string, l, r):
                while(l < r):
                    if(string[l] != string[r]):
                        return False
                    l += 1
                    r -= 1
                return True

    # Optimized Memoized
    def solve(string, leftIndex, rightIndex):
        
        if(leftIndex >= rightIndex):
            return 0 # already palindrome
        
        # New line
        if(DP[leftIndex][rightIndex] != -1):
            return DP[leftIndex][rightIndex]
        
        if(isPalindrome(string, leftIndex, rightIndex)):
            DP[leftIndex][rightIndex] = 0 # New Line
            return 0
        
        minAns = 2**31 - 1
        for k in range(leftIndex, rightIndex):
            # New part
            leftVal = 0
            rightVal = 0
            if(DP[leftIndex][k] != -1):
                leftVal = DP[leftIndex][k]
            else:
                DP[leftIndex][k] = solve(string, leftIndex, k)
                leftVal = DP[leftIndex][k]
            if(DP[k+1][rightIndex] != -1):
                rightVal = DP[k+1][rightIndex]
            else:
                DP[k+1][rightIndex] = solve(string, k+1, rightIndex)
                rightVal = DP[k+1][rightIndex]
            # New line
            tempAns = leftVal + rightVal + 1
            
            minAns = min(minAns, tempAns)
        
        DP[leftIndex][rightIndex] = minAns
        return DP[leftIndex][rightIndex]

    return solve(str, 0, len(str)-1)

    ## Boolean Paranthesis / Evaluate Expression to True
    # Recursive
    def solve(string, i, j, BooleanNeed):
        # base case
        if(j < i):
            return 0
        # base case
        if(i == j):
            if(BooleanNeed):
                return 1 if (string[i] == 'T') else 0
            else:
                return 1 if (string[i] == 'F') else 0
        
        ans = 0
        for k in range(i+1, j, 2):
            # tempAns
            leftFalse = solve(string, i, k-1, False)
            leftTrue = solve(string, i, k-1, True)
            rightFalse = solve(string, k+1, j, False)
            rightTrue = solve(string, k+1, j, True)
            
            # Operator wise conditions
            if(string[k] == '&'):
                if(BooleanNeed):
                    ans = ans + leftTrue*rightTrue
                else:
                    ans = ans + leftFalse*rightTrue + leftTrue*rightFalse + leftFalse*rightFalse
            elif(string[k] == '|'):
                if(BooleanNeed):
                    ans = ans + leftFalse*rightTrue + leftTrue*rightFalse + leftTrue*rightTrue
                else:
                    ans = ans + leftFalse*rightFalse
            elif(string[k] == '^'):
                if(BooleanNeed):
                    ans = ans + leftFalse*rightTrue + leftTrue*rightFalse
                else:
                    ans = ans + leftTrue*rightTrue + leftFalse*rightFalse
        
        return ans
    print(solve(string, 0, size - 1, True))

    # DP (Memoized) - O(n^3) and O(n^2) - why because defined 2 spaces true and false
    def solve(string, i, j, BooleanNeed):
        # base case
        if(j < i):
            return 0
        # New line
        if(BooleanNeed):
            if(DP[i][j][0] != -1):
                return DP[i][j][0]
        else:
            if(DP[i][j][1] != -1):
                return DP[i][j][1]
        # base case
        if(i == j):
            if(BooleanNeed): # True means 1st element of Array
                if (string[i] == 'T'):
                    DP[i][j][0] = 1
                else:
                    DP[i][j][0] = 0
                return DP[i][j][0]
            else:
                if(string[i] == 'F'):
                    DP[i][j][1] = 1
                else:
                    DP[i][j][1] = 0
                return DP[i][j][1]
            
        ans = 0
        for k in range(i+1, j, 2):
            # tempAns -- New Line slightly more optimized
            if(DP[i][k-1][1] == -1):
                DP[i][k-1][1] = solve(string, i, k-1, False)
            if(DP[i][k-1][0] == -1):
                DP[i][k-1][0] = solve(string, i, k-1, True)
            if(DP[k+1][j][1] == -1):
                DP[k+1][j][1] = solve(string, k+1, j, False)
            if(DP[k+1][j][0] == -1):
                DP[k+1][j][0] = solve(string, k+1, j, True)
            
            leftFalse = DP[i][k-1][1]
            leftTrue = DP[i][k-1][0]
            rightFalse = DP[k+1][j][1]
            rightTrue = DP[k+1][j][0]
            
            # Operator wise conditions
            if(string[k] == '&'):
                if(BooleanNeed):
                    ans = ans + leftTrue*rightTrue
                else:
                    ans = ans + leftFalse*rightTrue + leftTrue*rightFalse + leftFalse*rightFalse
            elif(string[k] == '|'):
                if(BooleanNeed):
                    ans = ans + leftFalse*rightTrue + leftTrue*rightFalse + leftTrue*rightTrue
                else:
                    ans = ans + leftFalse*rightFalse
            elif(string[k] == '^'):
                if(BooleanNeed):
                    ans = ans + leftFalse*rightTrue + leftTrue*rightFalse
                else:
                    ans = ans + leftTrue*rightTrue + leftFalse*rightFalse
        # New line
        if(BooleanNeed):
            DP[i][j][0] = ans
            return DP[i][j][0]
        else:
            DP[i][j][1] = ans
            return DP[i][j][1]

    print(solve(string, 0, size - 1, True)%1003)

    ## Scrambled Strings -- V tough

    ## Egg dropping problem
    # Recursion
    def solve(floors, eggs):
        # base case
        if(floors == 0 or floors == 1):
            return floors
        if(eggs == 1):
            return floors
        
        minAttempts = 2**31 - 1
        
        for k in range(1, floors + 1):
            tempAns = 1 + max(solve(floors - k,eggs), solve(k - 1, eggs - 1))
            minAttempts = min(minAttempts, tempAns)
        
        return minAttempts

    print(solve(floors, eggs))

    # Memoized
    def solve(floors, eggs):
        # base case
        if(floors == 0 or floors == 1):
            return floors
        if(eggs == 1):
            return floors
        # New line
        if(DP[eggs][floors] != -1):
            return DP[eggs][floors]
            
        minAttempts = 2**31 - 1
        
        for k in range(1, floors + 1):
            tempAns = 1 + max(solve(floors - k,eggs), solve(k - 1, eggs - 1))
            minAttempts = min(minAttempts, tempAns)
        
        # New line
        DP[eggs][floors] = minAttempts
        return DP[eggs][floors]

    DP = [[-1 for x in range(0, floors + 1)] for y in range(0, eggs + 1)]
    print(solve(floors, eggs))

    # Memoized more optimized
    def solve(floors, eggs):
        # base case
        if(floors == 0 or floors == 1):
            return floors
        if(eggs == 1):
            return floors
        if(DP[eggs][floors] != -1):
            return DP[eggs][floors]
            
        minAttempts = 2**31 - 1
        
        for k in range(1, floors + 1):
            # More optimization
            if(DP[eggs][floors-k] == -1):
                DP[eggs][floors-k] = solve(floors - k,eggs)
            if(DP[eggs-1][k-1] == -1):
                DP[eggs-1][k-1] = solve(k - 1, eggs - 1)
                
            tempAns = 1 + max(DP[eggs][floors-k], DP[eggs-1][k-1])
            minAttempts = min(minAttempts, tempAns)
        
        DP[eggs][floors] = minAttempts
        return DP[eggs][floors]

    DP = [[-1 for x in range(0, floors + 1)] for y in range(0, eggs + 1)]
    print(solve(floors, eggs))

#### Tree DP -- revise everyday

    ## Tree Diameter - DP O(n)
    def DiaTree(root, res):
            
            # Base Condition
            if(root is None):
                return 0
            
            # Hypothesis
            l = DiaTree(root.left, res)
            r = DiaTree(root.right, res)
            
            # Overall result
            res[0] = max([res[0], l+r+1, 1 + max(l ,r)])
            
            return 1 + max(l ,r) # this is temp
        
    res = [-2**31 + 1]

    DiaTree(root, res)

    return res[0]

    ## Max path sum
        # no constrain of leaf to leaf
        res = [-2**31 + 1]
        
        def maxPathSum(root, res):
            
            if(root is None):
                return 0
            
            l = maxPathSum(root.left, res)
            r = maxPathSum(root.right, res)
            
            # we have to pass on temp 
            # max(l, r) is negative and root.val is less negative or ositive
            # then only pass root.val (leaving nodes below it)
            temp = max(max(l,r) + root.val, root.val)
            
            # Same concept
            ans = max(temp, l+r+root.val)
            res[0] = max(res[0], ans)
            
            return temp
        
        maxPathSum(root, res)
        
        return res[0]

    ## Max Path sum between 2 leaf nodes -- nearly same code
        def maxPathSum(root, res):
                
            if(root is None):
                return 0
            
            l = maxPathSum(root.left, res)
            r = maxPathSum(root.right, res)
            
            if(root.left is not None and root.right is not None):
                temp = max(l,r) + root.data
                
                # ans = max(temp, l + r + root.data) - doing this takes into consideration that from a leaf node to non leaf node can end up as result
                res[0] = max(res[0], l + r + root.data)
                
                return temp
            elif(root.left is None):
                return r + root.data
            else:
                return l + root.data
            
        res = [-2**31  + 1]

        maxPathSum(root, res)

        return res[0]

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


## Array Questions - Goldman Sachs

## Sort a stack with stacks function only

Stack: 11 2 32 3 41
Output: 41 32 11 3 2

def sortStackWithStackFuncs(stack):
        
    if(len(stack) != 0):
        
        topElement = stack.pop() # gets the last element
        
        sortStackWithStackFuncs(stack) # sort remaining
        
        insertInSortedStack(stack, topElement)

def insertInSortedStack(stack, topElement):
    
    if(len(stack) == 0 or stack[-1] >= topElement):
        
        stack.append(topElement)
    
    else:
        
        newTopElement = stack.pop()
        
        insertInSortedStack(stack, topElement)
        
        stack.append(newTopElement)

sortStackWithStackFuncs(s)

## Check if Anagram
# O(N) and O(N)
# given str1 and str2
count1 = [0 for x in range(26)]
count2 = [0 for x in range(26)]

for char in str1:
    count1[ord(char) - 97] += 1

for char in str2:
    count2[ord(char) - 97] += 1

Flag = True
for i in range(26):
    if(count1[i] != count2[i]):
        Flag = False
        break

if(Flag):
    print('YES')
else:
    print("NO")

# get sum of ord of all chars then subtract from other string as 97 to 122 has no difficulty to have commoness
count = 0
for char in str1:
    count += ord(char)

for char in str2:
    count -= ord(char)

if(count == 0):
    print("YES")
else:
    print("NO")
