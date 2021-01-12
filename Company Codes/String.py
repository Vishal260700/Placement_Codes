# Check for Anagram - 2 strings are anagram if they have same charecters

    testCases = int(input())

    while(testCases > 0):
        
        Queries = input().split(' ')
        
        str1 = Queries[0]
        str2 = Queries[1]
        
        count = 0
        
        for char in str1:
            count += ord(char)
        
        for char in str2:
            count -= ord(char)
        
        if(count == 0):
            print("YES")
        else:
            print("NO")
            
        testCases -= 1

# Remove duplicates from a string with its order intact

    # Smart way - O(n) and O(n)
    
    Dict = set()
    res = ""
    for i in range(0, len(S)):
        if(S[i] not in Dict):
            Dict.add(S[i])
            res += S[i]
    return res
    
    # O(nlogn) and O(n)
    Dict = {}
    
    for i in range(0, len(S)):
        if(S[i] not in Dict):
            Dict[S[i]] = [i]
    
    Dict = ''.join(list(sorted(Dict, key = lambda k:Dict[k][0])))
    
    return (Dict)


https://practice.geeksforgeeks.org/problems/recursively-remove-all-adjacent-duplicates/0
https://practice.geeksforgeeks.org/problems/longest-palindrome-in-a-string/0