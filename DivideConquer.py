
# Unique element in sorted Array

# Brute force - Bit manipulation - O(N)
res = nums[0]
for i in range(1, len(nums)):
    res = res ^ nums[i]
return res

# binary Search Optimized - O(logN)
low = 0
high = len(nums) - 1

while(low < high):
    
    mid = 2*((low + high)//4)
    
    if(nums[mid] == nums[mid+1]):
        low = mid + 2
    else:
        high = mid

return nums[low]

# Find in a rotated Array

# Linear Search - O(N)
for i in range(0, len(nums)):
    if(nums[i] == target):
        return i
return -1

# Binary Search - O(logN)
low = 0
high = len(nums)-1

while(low <= high):
    
    mid = (low + high)//2
    
    # Jackpot
    if(nums[mid] == target):
        return mid
    
    # low to mid is sorted
    if(nums[low] <= nums[mid]):
        if(nums[mid] < target or nums[low] > target): # or statement enables that may be pivot is yet to be find and is at the lower side
            low = mid + 1
        else:
            high = mid - 1
    # mid to high is sorted
    else:
        if(nums[mid] > target or nums[high] < target): # or statement enables that may be pivot is yet to be find and is at the higher side
            high = mid - 1
        else:
            low = mid + 1

return -1

# Kth element of 2 sorted arrays - O(N) worst case
pointer1 = 0
pointer2 = 0

while(pointer1 < len(Arr1) and pointer2 < len(Arr2)):
    if(Arr1[pointer1] > Arr2[pointer2]):
        k -= 1
        if(k == 0):
            print(Arr2[pointer2])
            break
        pointer2 += 1
    else:
        k -= 1
        if(k == 0):
            print(Arr1[pointer1])
            break
        pointer1 += 1

if(k > 0):
    if(pointer1 < len(Arr1)):
        print(Arr1[pointer1 + k - 1])
    elif(pointer2 < len(Arr2)):
        print(Arr2[pointer2 + k - 1])

# Median of 2 sorted Arrays - O(N), time laga to get hold of even and odd number of total elements and figuring out their calculative way to get them
# like kth element here k = total len // 2 + 1
pointer1 = 0
pointer2 = 0

total = len(nums1) + len(nums2)

if(total % 2 != 0):
    k = (len(nums1) + len(nums2))//2 + 1

    while(pointer1 < len(nums1) and pointer2 < len(nums2)):
        if(nums1[pointer1] > nums2[pointer2]):
            k -= 1
            if(k == 0):
                return (nums2[pointer2])
            pointer2 += 1
        else:
            k -= 1
            if(k == 0):
                return (nums1[pointer1])
            pointer1 += 1

    if(k > 0):
        if(pointer1 < len(nums1)):
            return (nums1[pointer1 + k - 1])
        elif(pointer2 < len(nums2)):
            return (nums2[pointer2 + k - 1])
else:
    n1 = None
    n2 = None
    k = (len(nums1) + len(nums2))//2 + 1

    while(pointer1 < len(nums1) and pointer2 < len(nums2)):
        if(nums1[pointer1] > nums2[pointer2]):
            k -= 1
            if(k == 1):
                n1 = (nums2[pointer2])
            if(k == 0):
                n2 = (nums2[pointer2])
                break
            pointer2 += 1
        else:
            k -= 1
            if(k == 1):
                n1 = (nums1[pointer1])
            if(k == 0):
                n2 = (nums1[pointer1])
                break
            pointer1 += 1
    
    if(k > 0):
        if(pointer2 < len(nums2)):
            if(n1 is None):
                n1 = nums2[pointer2 + k - 1]
                n2 = nums2[pointer2 + k - 2]
            else:
                n2 = nums2[pointer2 + k - 1]
        else:
            if(n1 is None):
                n1 = nums1[pointer1 + k - 1]
                n2 = nums1[pointer1 + k - 2]
            else:
                n2 = nums1[pointer1 + k - 1]
    
    return (float(n1)+float(n2))/2

# Above median method with binary search - CHALLENGE
# Binary search way
# if we div both in 2 sets a1,a2 and b1,b2 s.t.
# a1 + b1 == a2 + b2 in length
# and elements of a1 + b1 <= elements of a2 and b2

# Binary search way -- GAJAB
# if we div both in 2 sets a1,a2 and b1,b2 s.t.
# a1 + b1 == a2 + b2 in length
# and elements of a1 + b1 <= elements of a2 and b2

# get lengths of given arrays
nums1_len = len(nums1)
nums2_len = len(nums2)

# Ensure nums2 is larger in size
if(nums1_len > nums2_len):
    return self.findMedianSortedArrays(nums2, nums1)

# Binary Search on nums1 i.e. smaller one
low = 0
high = nums1_len

while(low <= high):
    # partition lengths
    partition_len1 = (low + high)//2
    partition_len2 = (nums1_len + nums2_len + 1)//2 - partition_len1
    
    # a1 ka max and a2 ka min
    a1Max = nums1[partition_len1 - 1] if partition_len1 else -float('inf')
    a2Min = nums1[partition_len1] if partition_len1 < nums1_len else float('inf')
    
    # b1 ka max and b2 ka min
    b1Max = nums2[partition_len2 - 1] if partition_len2 else -float('inf')
    b2Min = nums2[partition_len2] if partition_len2 < nums2_len else float('inf')
    
    if(a1Max <= b2Min and b1Max <= a2Min):
        # Jackpot
        # Even
        if((nums1_len + nums2_len) % 2 == 0):
            return float((max(a1Max, b1Max) + min(b2Min, a2Min)))/2
        # Odd
        else:
            return float(max(a1Max, b1Max))
    # Binary Search moving forward
    elif(a1Max > b2Min):
        high = partition_len1 - 1
    else:
        low = partition_len1 + 1

## STRING QUESTION -- Reverse words in a string
Arr = s.split()
size = len(Arr)
for i in range(0, size//2):
    Arr[i], Arr[size - 1 - i] = Arr[size - 1 - i], Arr[i]   
return ' '.join(Arr)
