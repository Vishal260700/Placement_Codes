## Second Order Difference with missing values

# Each row in Arr is a stock with h stepsize
Arr = [[1, None, 4, None, 2, None, 6], 
       [9, None, None, None, 4, None, None, None, 3],
       [6, None, None, None, 1, None, None, None, 6]]

# helper function
def getStepSize(arr): # get h
    h = 0
    lastelem = None
    
    for i in range(0, len(arr)):
        if(lastelem and arr[i]):
            return h
        elif(arr[i] and not lastelem):
            lastelem = arr[i]
        h += 1 

# Cal Profit Arr of Stock
# Eq to implement = h^(-2)*(f(x) - 2*f(x-h) + f(x-2*h))
def cal(arr):
    # Step Size
    h = getStepSize(arr)
    stack = []
    res = []
    for i in range(0, len(arr)):
        if(arr[i]):
            if(len(stack) < 2):
                stack.append(arr[i])
                res.append(None)
            else:
                lastelem = stack.pop()
                lastTolastElem = stack.pop()
                temp = (h**(-2))*(arr[i] - 2*lastelem + lastTolastElem)
                stack.append(lastelem)
                stack.append(arr[i])
                res.append(temp)
        else:
            res.append(None)
    return res
    
for i in range(len(Arr)):
    print(cal(Arr[i]))

## Compute Industry Average Stock Return

# columns are days
# rows are stocks
Industry = [[2, 3, 2, 3, 2],
            [2, 1, 2, 1, 1],
            [3, 2, 3, 1, 1]]
ret = [[0.81472, 0.91338, 0.2785, 0.96489, 0.9571],
       [0.90579, 0.63236, 0.54688, 0.15761, 0.4851],
       [0.12699, 0.09754, 0.95751, 0.97059, 0.8001]]

res = []
for stock in range(0, len(ret[0])):
    DayRetData = []
    DayIndData = []
    
    for day in range(0, len(ret)):
        DayRetData.append(ret[day][stock])
        DayIndData.append(Industry[day][stock])
    
    StocksInd = {}
    for i in range(0, len(DayIndData)):
        StocksInd[i] = DayIndData[i]
    
    IndRetVal = {} # Key -> Industry Index, Value -> Return Value of that industry
    for i in range(0, len(DayRetData)):
        if(DayIndData[i] not in IndRetVal):
            IndRetVal[DayIndData[i]] = [DayRetData[i], 1]
        else:
            val = IndRetVal[DayIndData[i]][0] + DayRetData[i]
            count = IndRetVal[DayIndData[i]][1] + 1
            IndRetVal[DayIndData[i]] = [val, count]
    
    tempRes = []
    for i in range(0, len(DayIndData)):
        currInd = StocksInd[i]
        currRet = IndRetVal[currInd][0]/IndRetVal[currInd][1]
        tempRes.append(currRet)
    res.append(tempRes)

print(res)

## Count Substrings in a large passage
   
# Use this or string.count('th')
def countSub(string):
    count = 0    
    for i in range(0, len(string)-1):
        if(string[i] == 't' and string[i+1] == 'h'):
            count += 1
    return count
 
    