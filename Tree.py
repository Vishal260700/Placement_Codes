
## Height of Tree - O(N)
if(root is None):
    return 0
    
return 1 + max(height(root.left), height(root.right))

## Diameter of Tree - O(N), O(N)

    # 3 cases - 
    # 1) left side ke subtree mein max dia ho
    # 2) right side ke subtree mein max dia ho
    # 3) parent ko include karke max dia ho
    
    self.ans = 1
    
    def getMaxDia(root):
        
        # Leaf node
        if(not root):
            return 0
        
        L = getMaxDia(root.left) # left side
        R = getMaxDia(root.right) # right side
        
        self.ans = max(self.ans, L + R + 1) # overall answer to our question, +1 is to include the parent node
        
        return max(L, R) + 1 # return the value of this sub tree under question
    
    getMaxDia(root)
    
    return self.ans - 1

## 2 identical Binary Trees

def check(root1, root2):
    if(root1 is None and root2 is None):
        return 1
    elif(root1 is None and root2 is not None):
        return 0
    elif(root1 is not None and root2 is None):
        return 0
    elif(root1.data == root2.data):
        leftCheck = check(root1.left, root2.left)
        rightCheck = check(root1.right, root2.right)
        return (leftCheck and rightCheck)
    else:
        return 0
        
return check(root1, root2)

## Symmetric Binary tree check -- parent wise checking

if(root is None):
    return True

def check(parent1, parent2):
    
    if(parent1 is None and parent2 is None):
        return True
    elif(parent1 is not None and parent2 is not None):
        if(parent1.val == parent2.val):
            return check(parent1.left, parent2.right) and check(parent1.right, parent2.left)
        else:
            return False

return check(root.left, root.right)

## check if 2 tress are mirror or not

def areMirror(root1, root2):
    
    if(root1 is None and root2 is None):
        return True
    
    if(root1 is None or root2 is None):
        return False
    
    if(root1.val == root2.val):
        return (areMirror(root1.left, root2.right) and areMirror(root1.right, root2.left))
    
    return False

## Iterative of above
# Perform inorder and reverse inorder traversal of respective trees and compare respective nodes
def areMirrors(root1, root2): 
    res1 = [] 
    res2 = [] 
    while (True): 
          
        # iterative inorder traversal of 1st tree  
        # and reverse inoder traversal of 2nd tree  
        while (root1 and root2): 
              
            
            if (root1.data != root2.data):  
                return False
                  
            st1.append(root1)  
            st2.append(root2)  

            # check for left of 1st and right of 2nd
            root1 = root1.left  
            root2 = root2.right 
          
        # one exist and another doesnot and vice-versa
        if (not (root1 == None and root2 == None)):  
            return False
              
        if (not len(st1) == 0 and not len(st2) == 0): 
            root1 = st1[-1]  
            root2 = st2[-1]  
            st1.pop(-1)  
            st2.pop(-1)  
              
            # checkk for right of 1st and left of 2nd
            root1 = root1.right  
            root2 = root2.left 
        
        else: 
            break
      
    # After complete traversal
    return True

# Check key in BST


    # Iterative O(H), O(1)
    while(root):
        if(root.val == val):
            return root
        elif(root.val > val):
            root = root.left
        else:
            root = root.right
    return None
    
    
    # Recursive - O(H), O(H) , H -> height of tree
    def search(root, target):
        
        # Not present
        if(root is None):
            return None
        
        # Recursive checking
        if(root.val == target):
            return root
        elif(root.val > target):
            return search(root.left, target)
        else:
            return search(root.right, target)
    
    return search(root, val)
    

        
# chec if tree is height balanced or not   
    # Recursive O(H^2) and O(H)
    def getHeight(root):
        
        if(root is None):
            return 0
        
        lDepth = getHeight(root.left)
        rDepth = getHeight(root.right)
        
        if(lDepth > rDepth):
            return lDepth + 1
        else:
            return rDepth + 1
    
    def check(root):
        
        if(root is None):
            return True
        
        if(abs(getHeight(root.left) - getHeight(root.right)) <= 1):
            return check(root.left) and check(root.right)
        
        return False
    
    return check(root)

    # Optimized Recursive - do both functions together
    def check(root):
        # obvious base case
        if(root is None):
            return 1
        
        # check left right sub tree are balanced or not
        l = check(root.left)
        r = check(root.right)
        
        if(not l):
            return 0
        
        if(not r):
            return 0
        
        # Overall holistic return value
        return abs(l - r) <= 1 and (max(l, r) + 1) # max side takes care of leaf nodes i.e. when both l and r are 0
        
    return check(root)

## Inorder Traversal

        # Iterative O(n), O(n)
        stack = []
        curr = root
        
        res = []
        
        while(True):
            if(curr is not None):
                stack.append(curr)
                curr = curr.left
            elif(stack):
                now = stack.pop()
                res.append(now.val)
                curr = now.right
            else:
                break
        
        return res
        
        
        # Recursive O(n), O(n)
        def inorder(root, res):
            
            if(root is None):
                return res
            
            inorder(root.left, res)
            
            res.append(root.val)
            
            inorder(root.right, res)
            
            return res
        
        return inorder(root, [])

## Preorder Traversal

        # Iterative
        stack = []
        curr = root
        res = []
        
        while(True):
            if(curr is not None):
                res.append(curr.val)
                stack.append(curr)
                curr = curr.left
            elif(stack):
                now = stack.pop()
                curr = now.right
            else:
                break
        return res
        
        # Recursive
        def preorder(root, res):
            
            if(root is None):
                return res
            
            res.append(root.val)
            
            preorder(root.left, res)
            
            preorder(root.right, res)
            
            return res
        
        return preorder(root, [])

# Postorder Traversal


        # Iterative -- 2 stack
        stack = []
        curr = root
        res = []
        
        while(True):
            
            if(curr is not None):
                stack.append(curr)
                res.append(curr.val)
                curr = curr.right
            elif(stack):
                now = stack.pop()
                curr = now.left
            else:
                break
        
        return list(reversed(res))
        
        # recursive
        def postorder(root, res):
            if(root is None):
                return res
            
            postorder(root.left, res)
            postorder(root.right, res)
            res.append(root.val)
            
            return res
        
        return postorder(root, [])

## Validate if BT is BST

        # Iterative
        stack = [(root, float('-inf'), float('inf'))]
        
        while(stack):
            
            curr, lower, upper = stack.pop()
            
            if(curr is None):
                continue
            
            if(curr.val <= lower or curr.val >= upper):
                return False
            
            stack.append((curr.left, lower, curr.val))
            stack.append((curr.right, curr.val, upper))
        
        return True
        
        
        
        # Recursive
        def check(root, lower = float('-inf'), upper = float('inf')):
            
            if(root is None):
                return True
            
            if(root.val <= lower or root.val >= upper):
                return False
            
            if(not check(root.left, lower, root.val)):
                return False
            if(not check(root.right, root.val, upper)):
                return False
            
            return True
        
        return check(root)

####################################################################################################################################################################

# Floor in BST
def floor(root, key) : 
    # Base Case
    if(root is None):
        return INT_MAX
    # Original Value
    if(root.data == key):
        return root.data
    # Go to left tree
    elif(root.data > key):
        return floor(root.left, key)
    # Go to right tree
    else:
        # Check for if right part is None, left part didn't needed this as we are already moving lower (left is smaller)
        if(root.right is None):
            return root.data
        return floor(root.right, key)

# Ceil in BST
def ceil(root, key) : 
    # Base Case
    if(root is None):
        return -1
    # If present    
    if(root.data == key):
        return root.data
    elif(root.data > key):
        # Lower part, we need to check
        if(root.left is None):
            return root.data
        if(root.left.data < key):
            return root.data
        return ceil(root.left, key)
    else:
        # Right part of tree which is obvious to be greater
        return ceil(root.right, key)

# Two Sum in BST - O(N) O(N)

    Dict = {}
    
    def findTarget(target, root):
        # Base Case
        if(root is None):
            return False
        
        temp = target - root.val
        if(temp in Dict):
            return True
        else:
            Dict[root.val] = 1
            if(findTarget(target, root.left)):
                return True
            else:
                return findTarget(target, root.right)
    
    return findTarget(k, root)

    # Other methods - Inorder Traversal and get all elements in a array and then do 2 sum on an array with 2 pointers

---------------------------------------------------------------------------------------------------------------------------
# Kth smallest element in BST

    # Very slow but acceptable
    # count elements in a subtree
    def countElements(root, count):  0


        if(root is None):
            return count
        else:
            return 1 + countElements(root.left, count) + countElements(root.right, count)
    
    def getElements(root, k):
        # Base case
        if(root is None):
            return 0
        
        lCount = countElements(root.left, 0)
        
        ElemCount = lCount + 1
        
        if(k == ElemCount):
            return root.val
        elif(k <= ElemCount):
            return getElements(root.left, k)
        else:
            return getElements(root.right, k - ElemCount)
    
    return getElements(root, k)

    # Very Fast
    stack = []
    while(True):
        while(root):
            stack.append(root)
            root = root.left
        root = stack.pop()
        k -= 1
        if(k == 0):
            return root.val
        root = root.right

# Kth largest element in BST
stack = []
while(True):
    while(root):
        stack.append(root)
        root = root.right
    root = stack.pop()
    k -= 1
    if(k == 0):
        return root.data
    root = root.left
    

## Level Order Traversal - Queue Implementation - Like BFS
def levelOrder( root ):
    if(root is None):
        return 
    
    Queue = []
    Queue.append(root)
    res = []
    
    while(len(Queue)):
        curr = Queue.pop(0)
        
        res.append(curr.data)
        
        if(curr.left is not None):
            Queue.append(curr.left)
        
        if(curr.right is not None):
            Queue.append(curr.right)
        
    return res



## Level Order Traversal in Spiral Form

# Base Case
if(root is None):
    return

# stack1 -> left to right and stack2 -> right to left
stack1 = []
stack2 = []

# Result
res = []

# Init -- according to zig-zag form requirement
stack2.append(root)

while(len(stack1) != 0 or len(stack2) != 0):
    temp = []
    while(len(stack1) != 0):
        curr = stack1.pop()
        # Result
        temp.append(curr.val)
        # First right then left
        if(curr.right):
            stack2.append(curr.right)
        if(curr.left):
            stack2.append(curr.left)
    res.append(temp)
    temp = []
    while(len(stack2) != 0):
        curr = stack2.pop()
        # Result
        temp.append(curr.val)
        # First left then right
        if(curr.left):
            stack1.append(curr.left)
        if(curr.right):
            stack1.append(curr.right)
    res.append(temp)

# Take out first and last empty stored lists
if(res[0] == []):
    res = res[1:]
if(res[-1] == []):
    res = res[:len(res)-1]
return res


## Left view of Tree

max_level = [0] # state management
level = 1
res = []

def getLeftView(root, level, max_level):
    if(root is None):
        return
    
    if(max_level[0] < level):
        res.append(root.data)
        max_level[0] = level
    
    getLeftView(root.left, level + 1, max_level)
    getLeftView(root.right, level + 1, max_level)

getLeftView(root, level, max_level)

return res

## Right View of Tree
max_level = [0]
    level = 1
    res = []
    
    def getRightView(root, level, max_level):
        
        if(root is None):
            return
        
        if(max_level[0] < level):
            res.append(root.data)
            max_level[0] = level
        
        # This part is only different -- flip the Binary tree and get leftview
        getRightView(root.right, level + 1, max_level)
        getRightView(root.left, level + 1, max_level)
    
    getRightView(root, level, max_level)
    
    return res

# Top View of BT

# Stores [Node Value, Its level] based on key = horizontal distance (i.e. width of tree -> 0 at node left mein -ve)
Dict = {}

# Obvious from above statement
horizontalDistance = 0
verticalDistance = 0

def getTopView(root, horizontalDistance, verticalDistance, Dict):
    if(root is None):
        return
    
    # Unique top view with a element
    if(horizontalDistance not in Dict):
        Dict[horizontalDistance] = [root.data, verticalDistance] # verticalDistance is level
    # Not Unique and vertical distance is less of the present node then earlier stored one (right side wala abb aa raha ho)
    elif(verticalDistance < Dict[horizontalDistance][1]):
        Dict[horizontalDistance] = [root.data, verticalDistance] # Update at same horizontal level with new level and data
    
    # First left all then subsequently right ones
    getTopView(root.left, horizontalDistance - 1, verticalDistance + 1, Dict) # hor dist -1 
    getTopView(root.right, horizontalDistance + 1, verticalDistance + 1, Dict) # hor dist +1

getTopView(root, horizontalDistance, verticalDistance, Dict)

# based on sorted order -width/2 to width/2 (left to right)
for key in sorted(Dict.keys()):
    print(Dict[key][0], end = " ")

# Bottom View of BT

# Nearly same code except - *
Dict = {}
horizontalDistance = 0
verticalDistance = 0

def getBottomView(root, horizontalDistance, verticalDistance, Dict):
    if(root is None):
        return
    
    if(horizontalDistance not in Dict):
        Dict[horizontalDistance] = [root.data, verticalDistance]
    elif(verticalDistance > Dict[horizontalDistance][1]): # Change here (Obvious to look for)
        Dict[horizontalDistance] = [root.data, verticalDistance]
    
    # Most daunting change -- go right first -- right side visualise 
    #   2
    #  / \
    # 1   3
    # From top we want 3 to be seen so we go right first then left, and vice-veras in bottom view
    getBottomView(root.right, horizontalDistance + 1, verticalDistance + 1, Dict)
    getBottomView(root.left, horizontalDistance - 1, verticalDistance + 1, Dict)
    

getBottomView(root, horizontalDistance, verticalDistance, Dict)

res = []

for key in sorted(Dict.keys()):
    res.append(Dict[key][0])

return res

## LCA of 2 nodes in BT - O(n)
def lca(root, n1, n2):
    
    # Base Case
    if(root is None):
        return None
    # Found one of them
    if(root.data == n1 or root.data == n2):
        return root
    
    # Hypothesis -- kind of simple search
    leftLCA = lca(root.left, n1, n2)
    rightLCA = lca(root.right, n1, n2)
    
    # Main Implementation

    # Both present
    if(leftLCA and rightLCA):
        return root
    # Any one present i.e. ek hi side dono hai
    return leftLCA if leftLCA is not None else rightLCA

## LCA of 2 nodes in BST
# O(n)
def lca(root, n1, n2):
    
    # Base Case
    if(root is None):
        return None
    # Found one of them
    if(root.data == n1 or root.data == n2):
        return root
    
    # Hypothesis -- kind of simple search
    leftLCA = lca(root.left, n1, n2)
    rightLCA = lca(root.right, n1, n2)
    
    # Main Implementation

    # Both present
    if(leftLCA and rightLCA):
        return root
    # Any one present i.e. ek hi side dono hai
    return leftLCA if leftLCA is not None else rightLCA
# O(h)
# recursive
def LCA(root, n1, n2):
    #code here.
    
    # Base Case
    if(root is None):
        return None
    
    # This is Hypothesis + Main Algo
    # If root element is higher then both given then either this is ans or left part
    if(root.data > n1 and root.data > n2):
        return LCA(root.left, n1, n2)
    # Opposite happends here
    if(root.data < n1 and root.data < n2):
        return LCA(root.right, n1, n2)
    # If one of them are equal or there is a deadlock can't go either way i.e. ek bada dusra chota so only common is this
    return root
# Iterative -- like Recursive bas iterative kar diya
def LCA(root, n1, n2):
    
    while(root):
        
        if(root.data > n1 and root.data > n2):
            root = root.left
            continue
        
        if(root.data < n1 and root.data < n2):
            root = root.right
            continue
        
        return root
    
    return None

## Invert Binary Tree

parent = root
def reverseTree(parent):
    
    if(parent is None):
        return
    
    parent.left, parent.right = parent.right, parent.left
    
    reverseTree(parent.left)
    reverseTree(parent.right)

reverseTree(parent)
return root

## Add 2 trees
# O(N) and O(N)
def formTree(res, t1, t2):
            
    if(t1 is None and t2 is None):
        res = None
    elif(t1 is None or t2 is None):
        if(t1 is None):
            res = TreeNode(t2.val)
            res.left = formTree(res.left, None, t2.left)
            res.right = formTree(res.right, None, t2.right)
        else:
            res = TreeNode(t1.val)
            res.left = formTree(res.left, t1.left, None)
            res.right = formTree(res.right, t1.right, None)
    else:
        res = TreeNode(t1.val + t2.val)
        res.left = formTree(res.left, t1.left, t2.left)
        res.right = formTree(res.right, t1.right, t2.right)
    
    return res

return formTree(TreeNode(None), t1, t2)

# O(N) and O(1)
def mergeTrees(self, t1, t2):
    
    if(t1 is None):
        return t2
    if(t2 is None):
        return t1
    
    t1.val += t2.val
    t1.left = self.mergeTrees(t1.left, t2.left)
    t1.right = self.mergeTrees(t1.right, t2.right)
    
    return t1

## Level Order traversal - levle wise array arrangement
    3
   / \
  9  20
    /  \
   15   7

[
  [3],
  [9,20],
  [15,7]
]

# O(h)
def treeHeight(root):
            
    if(root is None):
        return 0
    
    leftDepth = treeHeight(root.left)
    rightDepth = treeHeight(root.right)
    
    if(leftDepth > rightDepth):
        return 1 + leftDepth
    else:
        return 1 + rightDepth

def levelOrderTraversal(root, level):
    if(root is None):
        return
    
    res[level].append(root.val)
    
    levelOrderTraversal(root.left, level+1)
    levelOrderTraversal(root.right, level+1)

res = [[] for x in range(treeHeight(root))]

levelOrderTraversal(root, 0)

return res
....

## Check BST
def checkBST(root, lower, higher):
        
    # None node is also a BST
    if(root is None):
        return True
    
    # leaf nodes always BST
    if(root.left is None and root.right is None):
        if(root.data > lower and root.data < higher):
            return True
        return False
    
    if(root.data > lower and root.data < higher):
        left = checkBST(root.left, lower, root.data)
        right = checkBST(root.right, root.data, higher)
        return left and right
    else:
        return False

return checkBST(root, -2**31 + 1, 2**31 - 1)

## Vertical Traversal of Tree
Dict = {} # key is vertical dist and value is array with upar wale in start
def TwoDTraversal(root, HorizontalDistance):

    if(root is None):
        return
    
    if(HorizontalDistance in Dict):
        Dict[HorizontalDistance].append(root.data)
    else:
        Dict[HorizontalDistance] = [root.data]
    
    TwoDTraversal(root.left, HorizontalDistance - 1)
    TwoDTraversal(root.right, HorizontalDistance + 1)

TwoDTraversal(root, 0)

res = []
for key in sorted(Dict):
    for nodeVal in Dict[key]:
        res.append(nodeVal)

return res

## Check for Balanced Tree
def balancedTree(root):
        
    if(root is None):
        return 0
    
    if(root.left is None and root.right is None):
        return 1
    
    left = balancedTree(root.left)
    right = balancedTree(root.right)
    
    if(left is -1 or right is -1):
        return -1
    
    if(abs(left - right) <= 1):
        return 1 + max(left, right)
    
    return -1

result = balancedTree(root)

if(result == -1):
    return False
return True