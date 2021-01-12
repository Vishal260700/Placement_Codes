Walmart Labs IIT Dhanbad 2018
Given strings A and B, find the minimum number of manipulation done in string A
to achieve the following:
1. A is a palindrome
2. B is a substring in A
Here manipulation is defined as changing a character to some other character
1 <= String Lengths <= 5000

SAMPLE INPUT:
7
xycdabyx abcd
acba abc
abcba abc
aaaa bbb
aa ab
aba c
abba c
SAMPLE OUTPUT:
6
-1
0
4
-1
1
2

# python sol - O(n^2)

# palindrome check - 2 pointers

low = 0
high = len(first) - 1

while(high >= low):
    if(first[low] == first[high]):
        low += 1
        high -= 1
    else:
        
