
# 1D List — building up

```Python
python# Basic accumulation
result = []
for i in range(5):
    result.append(i * i)
print(result)   # [0, 1, 4, 9, 16]
```

# Filtering while building

```Python
evens = []
for x in range(10):
    if x % 2 == 0:
        evens.append(x)
```

# Very common CP pattern: read n integers into a list

```Python
n = int(input())
arr = []
for _ in range(n):
    arr.append(int(input()))
```

# Or the idiomatic one-liner (same idea, more Pythonic)

```Python
arr = [int(x) for x in input().split()]
Stack usage (list as stack — append/pop from the end, O(1)):
pythonstack = []
stack.append(1)
stack.append(2)
stack.append(3)
top = stack.pop()      # 3, stack is now [1, 2]
Queue usage — DON'T use list.pop(0), it's O(n). Use collections.deque:
pythonfrom collections import deque
queue = deque()
queue.append(1)
queue.append(2)
front = queue.popleft()   # O(1), unlike list.pop(0)
2D List — building up (grids, adjacency matrices, DP tables)
python# Build empty grid, fill dynamically
rows, cols = 3, 4
grid = [[0] * cols for _ in range(rows)]   # remember: use list comprehension, not [[0]*cols]*rows!grid[1][2] = 5
```


# Build a 2D list row by row, from input

```Python
n, m = map(int, input().split())
grid = []
for _ in range(n):
    row = list(map(int, input().split()))
    grid.append(row)
```

# Growing a 2D structure dynamically (e.g. DP table filled as you compute)

```Python
dp = [[0] * (m+1) for _ in range(n+1)]
for i in range(1, n+1):
    for j in range(1, m+1):
        dp[i][j] = dp[i-1][j] + dp[i][j-1]   # e.g. path counting
Appending whole rows dynamically (when size isn't known upfront):
pythonresult = []
for i in range(n):
    row = []
    for j in range(m):
        if some_condition(i, j):
            row.append(1)
        else:
            row.append(0)
    result.append(row)
Dict — building up (very common: counting, graphs, memoization)
python# Frequency counting from scratch
freq = {}
for ch in "hello world":
    if ch not in freq:
        freq[ch] = 0
    freq[ch] += 1
```

# Cleaner: dict.get with default

```Python
freq = {}
for ch in "hello world":
    freq[ch] = freq.get(ch, 0) + 1
```

# Cleanest for CP: collections.Counter (does the whole thing in one line)

```Python
from collections import Counter
freq = Counter("hello world")
```


# Adjacency list (graph), built from edge list — VERY common in CP

```Python
from collections import defaultdict
graph = defaultdict(list)
edges = [(1,2), (2,3), (1,3)]
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)   # if undirected
```

# Memoization dict (DP), starts empty, filled as you recurse

```Python
memo = {}
def fib(n):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib(n-1) + fib(n-2)
    return memo[n]
defaultdict(list) is the CP workhorse for adjacency lists — it auto-creates an empty list the first time you touch a new key, so you skip the if key not in graph: graph[key] = [] boilerplate entirely.
Set — building up (visited nodes, deduplication)
python# Classic BFS/DFS visited set
visited = set()
visited.add(1)
visited.add(2)
if 3 not in visited:
    visited.add(3)
```

# Building a set from a loop (deduping)

```Python
seen = set()
unique_vals = []
for x in arr:
    if x not in seen:
        seen.add(x)
        unique_vals.append(x)
```

# Set of tuples — common for visited (row, col) pairs in grid BFS

```Python
visited = set()
visited.add((0, 0))
if (1, 0) not in visited:
    visited.add((1, 0))
Tuple — building up (a bit different since tuples are immutable)
You can't append to a tuple. The pattern instead is: build a list, then convert once you're done, OR build tuples incrementally as elements of a list/set.
python# Building a list of tuples (e.g. coordinate pairs, edges)
points = []
for i in range(3):
    for j in range(3):
        points.append((i, j))
```

# Then freeze if needed

```Python
points = tuple(points)   # if you want the whole collection immutable
Putting it together — a real CP-style BFS (uses several of these at once)
pythonfrom collections import deque, defaultdictdef bfs(start, graph):
    visited = set()
    visited.add(start)
    queue = deque([start])
    order = []    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)    return ordergraph = defaultdict(list)
edges = [(1,2), (1,3), (2,4), (3,4)]
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)print(bfs(1, graph))
```


This one function alone uses: empty set() built via .add(), empty deque() built via .append()/.popleft(), empty list built via .append(), and a defaultdict(list) built via .append() on auto-created keys.
