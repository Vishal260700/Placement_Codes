## LL done

# Swap Node in pairs
        
        # Recursive
        # Base Case
        if(head is None or head.next is None):
            return head
        # Main Algo (swappair wale mein 2 step ahead movement)
        head.next.next, head.next, head = head, self.swapPairs(head.next.next), head.next
        return head
        
        
        # Iterative
        # Lets have a dummy (Comb of 0 and head)
        dummy = ListNode(0)
        dummy.next = head
        
        # Prev and Curr
        prev = dummy
        curr = dummy.next
        
        # Main Algo -- VVImp
        while(curr and curr.next):
            curr.next.next, curr.next, prev.next = prev.next, curr.next.next, curr.next
            prev, curr = curr, curr.next
        
        return dummy.next


# Rotate a Linked List
        
        # Given head and k-shift to right
        if(head is None):
            return head
        # Traversing
        temp = head
        length = 1 # just before edge
        
        while(temp.next):
            length += 1
            temp = temp.next
        
        # Pattern ones out
        k = k%length
        if(k == 0):
            return head
        
        # Calculative iterations
        curr = head
        moves = length - k
        while(moves-1):
            curr = curr.next
            moves -= 1
        
        # we are now at edge of first part
        nextLeftover = curr.next
        curr.next = None
        
        # end of nextleftover to head i.e. the earlier temp is end of nextLeftover
        temp.next = head
        return nextLeftover

# Cycle in Linked List

        # Tortoise and hare method -- Fast and slow pointer
        slow = head
        fast = head
        
        while(slow and fast):
            if(fast.next is None):
                return False
            slow = slow.next
            fast = fast.next.next
            if(slow == fast):
                return True
        
        return False

# Remove Loop in LL -- find the node point at which loop starts/ends

        # Tortoise and hare method -- Fast and slow pointer
        slow = head
        fast = head
        
        while(fast and fast.next):
            slow = slow.next
            fast = fast.next.next
            if(slow == fast):
                slow = head
                while(slow != fast): # Imp
                    slow = slow.next
                    fast = fast.next # Imp
                return slow
        
        return None

# Merge 2 sorted LL

        # State management
        result = ListNode(None)
        res = result
        
        head1 = l1
        head2 = l2
        
        while(head1 and head2):
            if(head1.val < head2.val):
                # add head1
                res.next = ListNode(head1.val)
                res = res.next
                head1 = head1.next
            elif(head1.val == head2.val):
                # add both
                res.next = ListNode(head1.val)
                res = res.next
                res.next = ListNode(head1.val)
                res = res.next
                head1 = head1.next
                head2 = head2.next
            else:
                # add head2
                res.next = ListNode(head2.val)
                res = res.next
                head2 = head2.next
        
        if(head1):
            res.next = head1
        elif(head2):
            res.next = head2
        
        return result.next
    
# check if LL is palindrome

# Brute force - Array store all and traverse again pop em out if len arr == 0 return True - O(N), O(N)
# Optimized - O(N), O(1)
        
        #  reverse LL
        def rev(node):
            prev = None
            curr = node
            while(curr):
                newCurr = curr.next
                curr.next = prev
                prev = curr
                curr = newCurr
            return prev
       
        # get Length
        def LinkedLength(node):
            curr = node
            count = 0
            while(curr):
                count += 1
                curr = curr.next
            return count
        
        # Main Algo
        def solve(root):
            # length of LL
            length = LinkedLength(root)
            
            # get till mid of LL
            curr = root
            trav1 = length//2
            while(trav1):
                curr = curr.next
                trav1 -= 1
            
            # reverse remaining part of LL
            curr = rev(curr)
            
            # second traversal -- check them
            trav2 = length//2
            while(trav2):
                if(curr.val != root.val):
                    return False
                curr = curr.next
                root = root.next
                trav2 -= 1
            
            return True
        
        return solve(head)

## Delete a particular node in a LL which is given in starting and do all in place -- How is this O(1)
    ## Sol - val exhange karte ja and finally node.val = node.next.val i.e. None
    while(node):
        if(node.next.next is None):
            temp = node.val
            node.val = node.next.val
            node.next.val = temp
            node.next = node.next.next
            break
        else:
            temp = node.val
            node.val = node.next.val
            node.next.val = temp
            node = node.next

## Intersection of 2 LL

    # O(N), O(N)
    def lengthLL(root):
        curr = root
        count = 0
        while(curr):
            count += 1
            curr = curr.next
        return count
    
    l1 = lengthLL(headA)
    l2 = lengthLL(headB)
    
    if(l1 > l2):
        temp = l1 - l2
        while(temp):
            n = ListNode(0)
            n.next = headB
            headB = n
            temp -= 1
        while(l1):
            if(headA == headB):
                return headA
            headA = headA.next
            headB = headB.next
            l1 -= 1
    elif(l2 > l1):
        temp = l2 - l1
        while(temp):
            n = ListNode(0)
            n.next = headA
            headA = n
            temp -= 1
    
        while(l2):
            if(headA == headB):
                return headA
            headA = headA.next
            headB = headB.next
            l2 -= 1
    else:
        while(l1):
            if(headA == headB):
                return headA
            headA = headA.next
            headB = headB.next
            l1 -= 1
    return None

    # O(N) O(1) -- 2 pointers traverse each complete ince complete store last elemetn and redirect to other LL hence a lead of max(l1,l2) - min(l1,l2) is given to one 
    # thus they will reach some time and also with other conditions we can carry this out O(m+n)
    pointer1 = headA
    pointer2 = headB
    end1 = 0
    end2 = 0
    if(pointer1 is None or pointer2 is None):
        return None
    while(pointer1 or pointer2):
        # redirecting conditions
        if(pointer1 is None):
            pointer1 = headB
        if(pointer2 is None):
            pointer2 = headA
        
        # success conditions
        if(pointer1 == pointer2):
            return pointer1
        
        # Preperation for Unsuccessful condition
        if((pointer1.next is None) and (end1 == 0)):
            end1 = pointer1.val
        if((pointer2.next is None) and (end2 == 0)):
            end2 = pointer2.val
        
        # Unsuccess contions
        if((end1 != end2) and ((end1 != 0) and (end2 != 0))):
            return None
        
        pointer1 = pointer1.next
        pointer2 = pointer2.next
    
    return None

## Odd Even Linked List
# Input: 1->2->3->4->5->NULL
# Output: 1->3->5->2->4->NULL

# O(N) and O(N)
visited = []
returnList = ListNode(None)
res = returnList

curr = head
index = 1
while(curr):
    if(index%2 == 1):
        visited.append(0)
        temp = ListNode(curr.val)
        res.next = temp
        res = res.next
    else:
        visited.append(1)
    index += 1
    curr = curr.next

newCurr = head
newIndex = 1
while(newCurr):
    if(visited[newIndex-1] == 1):
        visited[newIndex-1] = 0
        temp = ListNode(newCurr.val)
        res.next = temp
        res = res.next
    newIndex += 1
    newCurr = newCurr.next

return returnList.next

# O(N) and O(1)
if (head is None):
    return head

odd = head
even = head.next
second_node = even

while(odd.next and even.next):
    odd.next = odd.next.next
    
    if(odd.next):
        odd = odd.next
    
    even.next = even.next.next
    even = even.next

odd.next = second_node


## 0s,1s,2s sort
temp = head
    
zeroes = 0
ones = 0
twos = 0

while(temp):
    if(temp.data == 0):
        zeroes += 1
    elif(temp.data == 1):
        ones += 1
    else:
        twos += 1
    temp = temp.next

temp2 = head

while(temp2):
    if(zeroes > 0):
        temp2.data = 0
        zeroes -= 1
    elif(ones > 0):
        temp2.data = 1
        ones -= 1
    else:
        temp2.data = 2
        twos -= 1
    temp2 = temp2.next

return head

# Delete without head pointer - maybe already done
temp = curr_node
while(temp.next.next):
    temp.data = temp.next.data
    temp = temp.next
temp.data = temp.next.data
temp.next = temp.next.next

# reverse a LL in groupd of K


## get nth node from end
# Two pointers
point1 = head
point2 = head

while(point2):
    if(n == 0):
        point1 = point1.next
    else:
        n -= 1
    point2 = point2.next

if(n != 0):
    return -1

return point1.data

## reverse LL in groups -- modified reversing
curr = head
prev = None
nextCurr = None
count = 0

while(curr and count < k):
    nextCurr = curr.next
    curr.next = prev
    prev = curr
    curr = nextCurr
    count += 1

if(nextCurr):
    head.next = reverse(nextCurr, k)

return prev

https://practice.geeksforgeeks.org/problems/flattening-a-linked-list/1
https://practice.geeksforgeeks.org/problems/implement-queue-using-linked-list/1