# Cartridge problem - O(N) Greedy
https://leetcode.com/discuss/interview-question/783699/recycling-cartridges-oa-hackerrank
cartridges = int(input())
dollars = int(input())
recycleReward = int(input())
perksCost = int(input())

perks = 0
while(cartridges):
    if(dollars < perksCost):
        dollars += recycleReward
        cartridges -= 1
    else:
        # these many we can make into perks
        potentialConversion = dollars//perksCost
        dollars = dollars - perksCost*potentialConversion
        cartridges = cartridges - potentialConversion
        perks += potentialConversion

print(perks)

# Same problem in DP - O(N^2)
cartridges = int(input())
dollars = int(input())
recycleReward = int(input())
perksCost = int(input())

# O(N^2)
DP = [[0 for x in range(0, dollars + 1)] for y in range(0, cartridges + 1)]
for y in range(0, cartridges + 1):
    for x in range(0, dollars + 1):
        if(x == 0 or y == 0):
            DP[y][x] = 0
        else:
            if(x + recycleReward <= dollars and x - perksCost >= 0):
                DP[y][x] = max(DP[y-1][x + recycleReward], DP[y-1][x - perksCost] + 1)
            elif(x + recycleReward <= dollars):
                DP[y][x] = DP[y-1][x+recycleReward]
            elif(x >= perksCost): # self explanable i.e. if given amt is greatr then perksCost
                DP[y][x] = DP[y-1][x-perksCost] + 1 
print(DP[-1][-1])


# Jump Games number of max steps we can take from each step given minimise steps to reach end
queue = []
visited = set()
queue.append((0, 0))
visited.add(0)

if(len(nums) <= 1):
    return 0

while(queue):
    curr, level = queue.pop(0)
    if(curr + nums[curr] >= len(nums) - 1):
        return level + 1
    else:
        for i in range(nums[curr], 0, -1):
            if(curr + i  not in visited):
                visited.add(curr + i)
                queue.append((curr + i, level + 1))
    
# Count Substrings that contain all vowels no consonants
# Python3 implementation of the approach 

# Function that returns true if c is a vowel 
def isVowel(c) : 

	return (c == 'a' or c == 'e' or c == 'i'
			or c == 'o' or c == 'u'); 


# Function to return the count of sub-strings 
# that contain every vowel at least 
# once and no consonant 
def countSubstringsUtil(s) : 

	count = 0; 

	# Map is used to store count of each vowel 
	mp = dict.fromkeys(s,0); 

	n = len(s); 

	# Start index is set to 0 initially 
	start = 0; 

	for i in range(n) : 
		mp[s[i]] += 1; 

		# If substring till now have all vowels 
		# atleast once increment start index until 
		# there are all vowels present between 
		# (start, i) and add n - i each time 
		while (mp['a'] > 0 and mp['e'] > 0
			and mp['i'] > 0 and mp['o'] > 0
			and mp['u'] > 0) : 
			count += n - i; 
			mp[s[start]] -= 1; 
			start += 1; 

	return count; 

# Function to extract all maximum length 
# sub-strings in s that contain only vowels 
# and then calls the countSubstringsUtil() to find 
# the count of valid sub-strings in that string 
def countSubstrings(s) : 

	count = 0; 
	temp = ""; 

	for i in range(len(s)) : 

		# If current character is a vowel then 
		# append it to the temp string 
		if (isVowel(s[i])) : 
			temp += s[i]; 

		# The sub-string containing all vowels ends here 
		else : 

			# If there was a valid sub-string 
			if (len(temp) > 0) : 
				count += countSubstringsUtil(temp); 

			# Reset temp string 
			temp = ""; 

	# For the last valid sub-string 
	if (len(temp) > 0) : 
		count += countSubstringsUtil(temp); 
	return count; 

# Driver code 
if __name__ == "__main__" : 
	s = "aeouisddaaeeiouua"; 
	print(countSubstrings(s)); 


## Prison Break
## https://snippets.cacher.io/snippet/e08f0fec1e61bf8f22a6

n = int(input())
m = int(input())

h = list(map(int, input().strip().split()))
v = list(map(int, input().strip().split()))

def solve(n, m, h, v):
    
    x = [1 for i in range(n + 1)]
    y = [1 for j in range(m + 1)]
    
    cX = 0
    cY = 0
    
    oX = -2**31 + 1
    oY = -2**31 + 1
    
    for i in range(0, len(h)):
        x[h[i]] = 0
    
    for j in range(0, len(v)):
        y[v[j]] = 0
    
    for i in range(1, n+1):
        if(x[i]):
            cX = 0
        else:
            cX += 1 
            oX = max(oX, cX)
    
    for j in range(1, m+1):
        if(y[j]):
            cY = 0
        else:
            cY += 1 
            oY = max(oY, cY)
    
    return ((oX+1) * (1+oY))

print(solve(n, m, h, v))
        

## Travelling is Fun -- https://leetcode.com/problems/graph-connectivity-with-threshold/
## Origin cities, target cities, divisors         
def connectedCities(n, g, originCities, destinationCities):
    if g == 0:
        return [1] * len(originCities)
    if g >= n:
        return [0] * len(originCities)
    components = {i: i for i in range(g + 1, n + 1)}
    for candidate in range(g + 1, n + 1):
        current = [t * candidate for t in range(1, n // candidate + 1)]
        current_components = [components[e] for e in current]
        minone = min(current_components)
        for curr in current:
            components[curr] = minone
    res = []
    for k in range(len(originCities)):
        origin, destination = originCities[k], destinationCities[k]
        if origin > g and destination > g and components[origin] == components[destination]:
            res.append(1)
        else:
            res.append(0)
    return res

## Scatter palindrome
https://leetcode.com/discuss/interview-question/431933/rubrik-oa-2019-scatter-palindrome

# Building Houses
# Bit Manipulation -- convert binary implementation to zero number of ways
# Inteligent String -- char value and atmost K with special chars
https://imgur.com/a/szcwQrG

# Min number of arrows to burst all ballons
https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/