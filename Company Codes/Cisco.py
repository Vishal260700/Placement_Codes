# given a grid of 3x3 and a condition og numbers swapable only if their sum is prime and they are adjacent
# swap them to make them linear 1, 2, 3, 4, 5, 6, 7, 8, 9
u = 3
grid = []
while(u):
    row = list(map(int, input().strip().split()))
    grid += row
    u -= 1

primes = [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1] # index based
swapAblePos = [[0,1], [0, 3], [1, 4], [1, 2], [2, 5], [3, 4], [3, 6], [4, 7], [4, 5], [5, 8], [6, 7], [7, 8]]
possiblities = {(1, 2, 3, 4, 5, 6, 7, 8, 9) : 0}
queue = [[1, 2, 3, 4, 5, 6, 7, 8, 9]]
while(queue):
    curr = queue.pop(0)
    for node in swapAblePos:
        x, y = node[0], node[1]
        if(primes[curr[x] + curr[y]] == 1):
            # sum is prime
            newGrid = curr[:]
            newGrid[x], newGrid[y] = newGrid[y], newGrid[x]
        if(tuple(newGrid) not in possiblities):
            possiblities[tuple(newGrid)] = possiblities[tuple(curr)] + 1
            queue.append(newGrid)

if(tuple(grid) in possiblities):
    print(possiblities[tuple(grid)])
else:
    print(-1)