# Coding interview training examples for model distillation.
# Each dict has a `prompt` (realistic interview-style problem) and an `answer`
# (a <thinking> section, production-grade code, and complexity/explanation notes).

examples = [
    {
        "prompt": "Given an array of integers `nums` and an integer `target`, return the indices of the two numbers that add up to `target`. You may assume exactly one solution exists, and you may not use the same element twice.",
        "answer": """<thinking>
The brute-force approach checks every pair, which is O(n^2). I can do better.

For each number `x`, the complement I need is `target - x`. If I store numbers I've already seen in a hash map (value -> index), then for each new number I can check in O(1) whether its complement was seen earlier. This gives a single pass.

Edge cases: duplicate values (e.g. [3, 3], target 6) work because I look up the complement before inserting the current element, so the earlier index is found.
</thinking>

```python
from typing import List


def two_sum(nums: List[int], target: int) -> List[int]:
    seen: dict[int, int] = {}
    for i, x in enumerate(nums):
        complement = target - x
        if complement in seen:
            return [seen[complement], i]
        seen[x] = i
    raise ValueError("No two sum solution exists")
```

**Complexity and Explanation**
- Time: O(n) - one pass over the array with O(1) hash lookups.
- Space: O(n) - the hash map may store up to n entries.

The key insight is trading space for time: instead of re-scanning the array for each element (O(n^2)), we remember what we've seen. Checking the complement *before* inserting the current element guarantees we never reuse the same index. An alternative is sorting plus two pointers (O(n log n) time, O(1) extra space if index tracking isn't needed), but that loses the original indices unless we store them first.""",
    },
    {
        "prompt": "Implement a function to reverse a singly linked list. The list node is defined with `val` and `next` attributes. Return the new head.",
        "answer": """<thinking>
Reversing a singly linked list means flipping each node's `next` pointer to point at its predecessor. I'll iterate with three references: `prev` (the already-reversed portion's head), `curr` (the node being processed), and a saved `next_node` so I don't lose the rest of the list.

The iterative approach uses O(1) extra space. A recursive version is elegant but uses O(n) stack space and risks overflow on long lists, so I prefer iterative for production.
</thinking>

```python
from typing import Optional


class ListNode:
    def __init__(self, val: int = 0, next: "Optional[ListNode]" = None):
        self.val = val
        self.next = next


def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    prev: Optional[ListNode] = None
    curr = head
    while curr is not None:
        next_node = curr.next  # save the rest of the list
        curr.next = prev       # reverse the pointer
        prev = curr            # advance prev
        curr = next_node       # advance curr
    return prev
```

**Complexity and Explanation**
- Time: O(n) - each node is visited exactly once.
- Space: O(1) - only a constant number of pointers.

We walk the list once, re-pointing each node backward. `prev` accumulates the reversed prefix and becomes the new head when `curr` falls off the end. The recursive alternative reverses the tail first then fixes the current link, but it costs O(n) call-stack space, making the iterative version safer for large inputs.""",
    },
    {
        "prompt": "Given the root of a binary tree, return its level-order traversal (a list of lists, where each inner list contains the node values at that depth, left to right).",
        "answer": """<thinking>
Level-order traversal is a breadth-first search. I'll use a queue. The trick to grouping by level is to record the queue's size at the start of each level, then process exactly that many nodes before moving on - everything enqueued during that batch belongs to the next level.

Edge case: an empty tree returns an empty list.
</thinking>

```python
from collections import deque
from typing import List, Optional


class TreeNode:
    def __init__(self, val: int = 0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def level_order(root: Optional[TreeNode]) -> List[List[int]]:
    if root is None:
        return []
    result: List[List[int]] = []
    queue: deque[TreeNode] = deque([root])
    while queue:
        level_size = len(queue)
        level: List[int] = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result
```

**Complexity and Explanation**
- Time: O(n) - every node is enqueued and dequeued once.
- Space: O(n) - the queue holds up to the widest level, which can be ~n/2 nodes.

Snapshotting `len(queue)` before each level is the core idea: it separates "current level" nodes from their children. A DFS variant can also produce level order by passing depth and indexing into `result[depth]`, but BFS maps most naturally to the problem.""",
    },
    {
        "prompt": "You are climbing a staircase with `n` steps. Each time you can climb either 1 or 2 steps. Return the number of distinct ways to reach the top.",
        "answer": """<thinking>
To reach step `n`, the last move was either from step n-1 (a 1-step) or step n-2 (a 2-step). So ways(n) = ways(n-1) + ways(n-2) - this is the Fibonacci recurrence.

Naive recursion is exponential. I'll use bottom-up DP. Since each state only depends on the previous two, I can keep just two rolling variables for O(1) space.

Base cases: 1 way to climb 0 steps (do nothing), 1 way for 1 step.
</thinking>

```python
def climb_stairs(n: int) -> int:
    if n <= 1:
        return 1
    prev2, prev1 = 1, 1  # ways to reach steps 0 and 1
    for _ in range(2, n + 1):
        prev2, prev1 = prev1, prev1 + prev2
    return prev1
```

**Complexity and Explanation**
- Time: O(n) - a single loop from 2 to n.
- Space: O(1) - only two scalar variables.

This is Fibonacci in disguise. The recurrence falls out of asking "what was my last step?" Rolling two variables avoids an O(n) DP array. For extremely large `n`, matrix exponentiation or Binet's formula reaches O(log n), but the linear solution is standard and overflow-safe in Python.""",
    },
    {
        "prompt": "Given a string `s`, find the length of the longest substring without repeating characters.",
        "answer": """<thinking>
This is a classic sliding-window problem. I maintain a window [left, right] containing only distinct characters. As I extend `right`, if I hit a character already in the window, I move `left` forward past its previous occurrence.

Using a dict that maps char -> last index lets me jump `left` directly instead of shrinking one step at a time. I must guard against moving `left` backward (only advance it).
</thinking>

```python
def length_of_longest_substring(s: str) -> int:
    last_seen: dict[str, int] = {}
    left = 0
    longest = 0
    for right, ch in enumerate(s):
        if ch in last_seen and last_seen[ch] >= left:
            left = last_seen[ch] + 1
        last_seen[ch] = right
        longest = max(longest, right - left + 1)
    return longest
```

**Complexity and Explanation**
- Time: O(n) - each character is visited once; `left` only moves forward.
- Space: O(min(n, m)) - where m is the charset size; the dict holds at most one entry per distinct character.

The window invariant is "all characters in [left, right] are unique." When a duplicate appears inside the window, we collapse the left edge just past the earlier copy. The `>= left` check prevents stale indices (characters last seen *before* the current window) from incorrectly pushing `left` backward.""",
    },
    {
        "prompt": "Given an m x n grid where each cell is '1' (land) or '0' (water), count the number of islands. An island is connected horizontally or vertically.",
        "answer": """<thinking>
Each island is a connected component of '1' cells. I'll scan the grid; when I find an unvisited '1', I increment the count and flood-fill the whole island so I don't count it again.

Flood fill can be DFS or BFS. I'll use an iterative BFS with a queue to avoid recursion depth limits on large grids. I'll mark visited cells by mutating them to '0' (or use a visited set if mutation isn't allowed).
</thinking>

```python
from collections import deque
from typing import List


def num_islands(grid: List[List[str]]) -> int:
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])
    count = 0

    def bfs(sr: int, sc: int) -> None:
        queue = deque([(sr, sc)])
        grid[sr][sc] = "0"
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == "1":
                    grid[nr][nc] = "0"  # mark before enqueue to avoid duplicates
                    queue.append((nr, nc))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1":
                count += 1
                bfs(r, c)
    return count
```

**Complexity and Explanation**
- Time: O(m*n) - each cell is examined a constant number of times.
- Space: O(min(m, n)) - the BFS queue's worst case for the largest level (DFS recursion would be O(m*n)).

We count connected components. Marking a cell '0' the moment we enqueue it (not when we dequeue) prevents the same cell being queued by two neighbors. If mutating the input is disallowed, swap the marking for a `visited` set at the cost of O(m*n) extra space.""",
    },
    {
        "prompt": "Merge two sorted linked lists into one sorted list and return its head. Splice the nodes of the original lists rather than creating new nodes.",
        "answer": """<thinking>
Standard two-pointer merge. I use a dummy head to simplify edge cases (no special handling for the first node). I walk both lists, attaching the smaller current node to the merged tail. When one list runs out, I attach the remainder of the other.

Using a sentinel/dummy node avoids null checks when building the result.
</thinking>

```python
from typing import Optional


class ListNode:
    def __init__(self, val: int = 0, next: "Optional[ListNode]" = None):
        self.val = val
        self.next = next


def merge_two_lists(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode()
    tail = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            tail.next = l1
            l1 = l1.next
        else:
            tail.next = l2
            l2 = l2.next
        tail = tail.next
    tail.next = l1 if l1 else l2  # attach whatever remains
    return dummy.next
```

**Complexity and Explanation**
- Time: O(n + m) - we advance through each list once.
- Space: O(1) - we relink existing nodes; only a constant number of pointers.

The dummy node is the trick that removes boilerplate: `dummy.next` ends up being the true head regardless of which list contributed the first element. Using `<=` (not `<`) keeps the merge stable, preserving relative order of equal values.""",
    },
    {
        "prompt": "Given an array `nums`, return an array `output` where `output[i]` is the product of all elements of `nums` except `nums[i]`. Solve it without using division and in O(n) time.",
        "answer": """<thinking>
Without division, I compute for each index the product of everything to its left and everything to its right, then multiply them.

Two passes: first a left-to-right pass storing prefix products, then a right-to-left pass multiplying in the suffix products. I can keep the output array doubling as the prefix store and use a single running suffix variable to hit O(1) extra space (excluding the output).
</thinking>

```python
from typing import List


def product_except_self(nums: List[int]) -> List[int]:
    n = len(nums)
    output = [1] * n
    # prefix products: output[i] = product of nums[0..i-1]
    prefix = 1
    for i in range(n):
        output[i] = prefix
        prefix *= nums[i]
    # multiply by suffix products: product of nums[i+1..n-1]
    suffix = 1
    for i in range(n - 1, -1, -1):
        output[i] *= suffix
        suffix *= nums[i]
    return output
```

**Complexity and Explanation**
- Time: O(n) - two linear passes.
- Space: O(1) extra - the output array is required output, not auxiliary space; only two scalars are extra.

The insight: result[i] = (product of all left elements) * (product of all right elements). We avoid division by building these prefix/suffix products explicitly. Reusing the output array for prefixes and folding the suffix in on the second pass is the space-optimal trick. This also sidesteps the division-by-zero problem that a naive total-product approach would face.""",
    },
    {
        "prompt": "Determine if a string `s` containing just the characters '()[]{}' is valid. Brackets must close in the correct order and each closing bracket must match the most recent unmatched opening bracket.",
        "answer": """<thinking>
This is the canonical stack problem. I push opening brackets. On a closing bracket, the top of the stack must be its matching opener; otherwise it's invalid. At the end the stack must be empty (no unclosed openers).

A dict mapping closer -> opener makes the matching clean. I also handle the case where a closer appears with an empty stack.
</thinking>

```python
def is_valid(s: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for ch in s:
        if ch in pairs:  # closing bracket
            if not stack or stack.pop() != pairs[ch]:
                return False
        else:  # opening bracket
            stack.append(ch)
    return not stack
```

**Complexity and Explanation**
- Time: O(n) - single pass; each character is pushed/popped at most once.
- Space: O(n) - the stack can hold up to n opening brackets (e.g. "(((((").

A stack naturally models "most recent unmatched opener must close first" (LIFO). The two failure modes are: a closer that doesn't match the top, and leftover openers at the end. Checking `not stack` before popping prevents an IndexError on inputs like "]".""",
    },
    {
        "prompt": "Given an array of `n` integers where each value is in the range [1, n] and exactly one number is missing (and one may be duplicated), find the single missing number when the array contains numbers 0..n with one missing. Use O(1) extra space.",
        "answer": """<thinking>
Classic "missing number" from 0..n with one absent. The sum of 0..n is n*(n+1)/2. Subtract the actual array sum to get the missing value. This is O(n) time, O(1) space, and avoids overflow concerns in Python.

An alternative is XOR: XOR all indices 0..n with all values; pairs cancel and the missing number remains. XOR avoids any overflow even in fixed-width languages.
</thinking>

```python
from typing import List


def missing_number(nums: List[int]) -> int:
    n = len(nums)
    expected = n * (n + 1) // 2
    return expected - sum(nums)


def missing_number_xor(nums: List[int]) -> int:
    result = len(nums)
    for i, x in enumerate(nums):
        result ^= i ^ x
    return result
```

**Complexity and Explanation**
- Time: O(n) - one pass to sum (or XOR).
- Space: O(1) - constant extra storage.

The sum approach uses Gauss's formula for 0..n and the fact that the difference is exactly the missing element. The XOR approach pairs each index with each value; identical numbers cancel (x ^ x = 0), leaving only the unmatched (missing) one. XOR is preferable in languages with fixed-width integers because it never overflows.""",
    },
    {
        "prompt": "Implement binary search: given a sorted array `nums` and a `target`, return the index of `target` or -1 if not present.",
        "answer": """<thinking>
Binary search repeatedly halves the search range. The two correctness pitfalls are the loop condition and the midpoint computation. I'll use a closed interval [lo, hi] with the condition `lo <= hi`, and compute mid as `lo + (hi - lo) // 2` to avoid integer overflow in languages where it matters.

I must update bounds correctly: when target is greater, search right (lo = mid + 1); when smaller, search left (hi = mid - 1). The `+1`/`-1` ensure progress and termination.
</thinking>

```python
from typing import List


def binary_search(nums: List[int], target: int) -> int:
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
```

**Complexity and Explanation**
- Time: O(log n) - the range halves each iteration.
- Space: O(1) - iterative, constant pointers.

The invariant is "if target exists, it lies within [lo, hi]." Using a closed interval with `lo <= hi` correctly handles the single-element case. The `mid +/- 1` updates guarantee the interval strictly shrinks, preventing infinite loops. Computing `lo + (hi - lo) // 2` instead of `(lo + hi) // 2` avoids overflow in fixed-width integer languages.""",
    },
    {
        "prompt": "Given the root of a binary tree, return its maximum depth (the number of nodes along the longest path from the root down to the farthest leaf).",
        "answer": """<thinking>
The depth of a tree is 1 + the max depth of its two subtrees, with an empty tree having depth 0. This recurrence gives a clean recursive solution.

Recursion depth equals tree height, which is fine for balanced trees but could overflow for degenerate (linked-list-like) trees. I'll present the clean recursive version and note an iterative BFS alternative.
</thinking>

```python
from typing import Optional


class TreeNode:
    def __init__(self, val: int = 0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_depth(root: Optional[TreeNode]) -> int:
    if root is None:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))
```

**Complexity and Explanation**
- Time: O(n) - each node is visited once.
- Space: O(h) - recursion stack proportional to tree height h (O(log n) balanced, O(n) worst case).

The recurrence "depth = 1 + max(left depth, right depth)" mirrors the tree's structure exactly. For very deep, unbalanced trees, an iterative level-order BFS counting levels avoids stack overflow while keeping O(n) time, trading recursion stack for an explicit queue.""",
    },
    {
        "prompt": "Given an integer array `nums`, find the contiguous subarray with the largest sum and return that sum (Kadane's algorithm).",
        "answer": """<thinking>
Kadane's algorithm: at each position, the best subarray ending here is either this element alone or this element extended onto the best subarray ending at the previous position. So current = max(x, current + x). Track the global maximum.

I must initialize with the first element (not 0) to handle all-negative arrays correctly - the answer should be the largest single element in that case.
</thinking>

```python
from typing import List


def max_subarray(nums: List[int]) -> int:
    best = current = nums[0]
    for x in nums[1:]:
        current = max(x, current + x)
        best = max(best, current)
    return best
```

**Complexity and Explanation**
- Time: O(n) - single pass.
- Space: O(1) - two running variables.

The DP insight: the maximum subarray ending at index i either starts fresh at i or extends the best one ending at i-1. We never need the full history, just the running `current`. Initializing both variables to `nums[0]` (rather than 0) is critical for arrays of all negatives, where the answer is the maximum element itself, not an empty subarray.""",
    },
    {
        "prompt": "Detect whether a singly linked list has a cycle. Return True if a cycle exists, False otherwise. Use O(1) extra space.",
        "answer": """<thinking>
Floyd's tortoise-and-hare: advance a slow pointer one step and a fast pointer two steps. If there's a cycle, the fast pointer eventually laps the slow one and they meet. If fast reaches the end (null), there's no cycle.

This uses O(1) space versus the O(n) hash-set approach of remembering visited nodes.
</thinking>

```python
from typing import Optional


class ListNode:
    def __init__(self, val: int = 0, next: "Optional[ListNode]" = None):
        self.val = val
        self.next = next


def has_cycle(head: Optional[ListNode]) -> bool:
    slow = fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False
```

**Complexity and Explanation**
- Time: O(n) - if a cycle exists, the fast pointer catches up within one loop traversal; otherwise it reaches the end in n/2 steps.
- Space: O(1) - just two pointers.

The two pointers close the gap by one node per iteration inside a cycle, guaranteeing a meeting. The loop guard `fast and fast.next` safely handles both even- and odd-length acyclic lists. The hash-set alternative is simpler to reason about but costs O(n) memory.""",
    },
    {
        "prompt": "Given an array `nums` sorted in non-decreasing order, remove duplicates in place such that each unique element appears once, and return the number of unique elements `k`. The first `k` elements should hold the unique values.",
        "answer": """<thinking>
Because the array is sorted, duplicates are adjacent. I use a slow/fast two-pointer approach: `slow` marks the position of the last unique element written; `fast` scans ahead. When `nums[fast]` differs from `nums[slow]`, it's a new unique value, so I advance `slow` and copy it there.

Handle the empty array up front.
</thinking>

```python
from typing import List


def remove_duplicates(nums: List[int]) -> int:
    if not nums:
        return 0
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1
```

**Complexity and Explanation**
- Time: O(n) - one pass with the fast pointer.
- Space: O(1) - in-place, no extra structures.

Sorted order is what makes this work: equal values cluster together, so a single comparison against the last kept element detects duplicates. `slow` always points to the end of the deduplicated prefix; `slow + 1` is therefore the count `k`. Everything past index k is irrelevant per the problem's contract.""",
    },
    {
        "prompt": "Given an array of intervals where intervals[i] = [start, end], merge all overlapping intervals and return the non-overlapping intervals that cover all the input intervals.",
        "answer": """<thinking>
Sort intervals by start time. Then sweep left to right: keep the last merged interval; if the current interval starts before (or at) the last one's end, they overlap, so extend the last interval's end to the max of the two ends. Otherwise, start a new merged interval.

Sorting is the enabling step - it guarantees that overlaps are only ever with the most recently merged interval.
</thinking>

```python
from typing import List


def merge_intervals(intervals: List[List[int]]) -> List[List[int]]:
    if not intervals:
        return []
    intervals.sort(key=lambda iv: iv[0])
    merged: List[List[int]] = [intervals[0][:]]
    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1]:           # overlap
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return merged
```

**Complexity and Explanation**
- Time: O(n log n) - dominated by the sort; the sweep is O(n).
- Space: O(n) - for the output (O(log n) to O(n) for sorting depending on implementation).

Sorting by start time means any interval that overlaps the current one must overlap the *last* merged interval specifically, reducing a quadratic pairwise check to a linear sweep. We extend by `max(last_end, end)` because a fully-contained interval shouldn't shrink the merged range.""",
    },
    {
        "prompt": "Given two strings `s` and `t`, return True if `t` is an anagram of `s` (same characters with the same frequencies), and False otherwise.",
        "answer": """<thinking>
Two strings are anagrams iff they have identical character frequency counts. Quick rejection: if lengths differ, return False immediately.

I'll count characters of `s`, decrement for `t`, and check all counts return to zero. Python's Counter makes this concise, and comparing two Counters is clean and clear.
</thinking>

```python
from collections import Counter


def is_anagram(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    return Counter(s) == Counter(t)
```

**Complexity and Explanation**
- Time: O(n) - building and comparing counts is linear in string length.
- Space: O(k) - where k is the number of distinct characters (O(1) for a fixed alphabet like lowercase English).

Anagram equivalence reduces to multiset equality of characters, which frequency counting captures exactly. The early length check is a cheap fast-path rejection. For a known small alphabet, a fixed-size array of 26 counts beats a hash map on constant factors; for Unicode, the Counter generalizes cleanly.""",
    },
    {
        "prompt": "Given `coins` of different denominations and a total `amount`, return the fewest number of coins needed to make up that amount. If it cannot be made, return -1. You have an infinite supply of each coin.",
        "answer": """<thinking>
This is the unbounded coin-change minimization, an unbounded knapsack variant. Greedy fails in general (e.g. coins [1,3,4], amount 6 - greedy picks 4+1+1=3 coins but 3+3=2 is optimal). So I use DP.

dp[a] = minimum coins to make amount a. dp[0] = 0. For each amount, dp[a] = min over coins c of dp[a - c] + 1. Initialize unreachable amounts to infinity.
</thinking>

```python
from typing import List


def coin_change(coins: List[int], amount: int) -> int:
    INF = amount + 1  # sentinel larger than any valid answer
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a:
                dp[a] = min(dp[a], dp[a - c] + 1)
    return dp[amount] if dp[amount] != INF else -1
```

**Complexity and Explanation**
- Time: O(amount * len(coins)) - we evaluate every coin at every sub-amount.
- Space: O(amount) - the 1D DP table.

Greedy is incorrect for arbitrary denominations, so we solve every sub-amount optimally and build up. dp[a] depends only on smaller amounts, making bottom-up iteration valid. Using `amount + 1` as the "infinity" sentinel is safe because no valid solution can use more than `amount` coins (each coin is at least 1).""",
    },
    {
        "prompt": "Given an array `nums` of size n, return the majority element - the element that appears more than n/2 times. You may assume it always exists. Solve in O(1) space.",
        "answer": """<thinking>
The Boyer-Moore voting algorithm finds a majority element in O(1) space. Maintain a candidate and a count. For each element: if count is 0, adopt the current element as candidate; then increment count if it matches the candidate, else decrement.

Because the majority element appears more than n/2 times, it cannot be fully canceled out by all other elements combined, so it survives as the final candidate.
</thinking>

```python
from typing import List


def majority_element(nums: List[int]) -> int:
    candidate = None
    count = 0
    for x in nums:
        if count == 0:
            candidate = x
        count += 1 if x == candidate else -1
    return candidate
```

**Complexity and Explanation**
- Time: O(n) - single pass.
- Space: O(1) - one candidate and one counter.

The intuition is pairing off: each non-majority element can cancel one majority occurrence, but since the majority exceeds n/2, it has more votes than everything else combined and ends as the surviving candidate. A hash-map frequency count also works in O(n) time but needs O(n) space; Boyer-Moore is the space-optimal answer.""",
    },
    {
        "prompt": "Count the number of 1 bits (Hamming weight) in the binary representation of an unsigned integer `n`.",
        "answer": """<thinking>
The elegant trick is Brian Kernighan's algorithm: `n & (n - 1)` clears the lowest set bit. Repeating until n is zero counts exactly as many iterations as there are set bits - so the loop runs only as many times as there are 1s, not 32/64 times.

This beats the naive bit-by-bit shift loop when the number of set bits is small.
</thinking>

```python
def hamming_weight(n: int) -> int:
    count = 0
    while n:
        n &= n - 1  # clears the lowest set bit
        count += 1
    return count
```

**Complexity and Explanation**
- Time: O(k) - where k is the number of set bits (at most the bit width).
- Space: O(1) - a single counter.

Subtracting 1 flips the lowest set bit to 0 and all bits below it to 1; ANDing with the original therefore removes exactly that lowest set bit. Each iteration eliminates one 1, so the loop count equals the popcount. This is faster than scanning all bits when few are set. Many CPUs expose this directly as a POPCNT instruction (Python's `int.bit_count()` in 3.10+).""",
    },
    {
        "prompt": "Given a binary search tree (BST) and two nodes, find their lowest common ancestor (LCA). You may assume both nodes exist in the tree.",
        "answer": """<thinking>
A BST's ordering makes LCA easy. Starting from the root: if both target values are smaller than the current node, the LCA is in the left subtree; if both are larger, it's in the right subtree. Otherwise the current node is the split point - it's the LCA (one target is on each side, or one equals the current node).

This is iterative and uses O(1) space.
</thinking>

```python
class TreeNode:
    def __init__(self, val: int = 0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def lowest_common_ancestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    node = root
    while node:
        if p.val < node.val and q.val < node.val:
            node = node.left
        elif p.val > node.val and q.val > node.val:
            node = node.right
        else:
            return node  # split point: p and q diverge here
    return root  # unreachable when both nodes exist
```

**Complexity and Explanation**
- Time: O(h) - we descend one path; h is tree height (O(log n) balanced, O(n) worst case).
- Space: O(1) - iterative traversal.

The BST property is the key: the LCA is the first node where the two targets fall on opposite sides (or one equals the node). Before that point both targets share a direction; after it they diverge. For a general (non-BST) binary tree we'd instead need a recursive post-order search, which is O(n).""",
    },
    {
        "prompt": "Given an array of strings, group the anagrams together. Return a list of groups; the order of groups and within groups does not matter.",
        "answer": """<thinking>
Anagrams share the same multiset of characters, so they share a canonical key. Two common keys: the sorted string, or a 26-length character-count tuple. Sorting each word is O(L log L); the count tuple is O(L) per word and avoids the log factor for long words.

I'll group by a frozen character-count key into a dict, then return the grouped values.
</thinking>

```python
from collections import defaultdict
from typing import List


def group_anagrams(strs: List[str]) -> List[List[str]]:
    groups: dict[tuple, List[str]] = defaultdict(list)
    for word in strs:
        counts = [0] * 26
        for ch in word:
            counts[ord(ch) - ord("a")] += 1
        groups[tuple(counts)].append(word)
    return list(groups.values())
```

**Complexity and Explanation**
- Time: O(n * L) - n words, each of length up to L, building a count key in linear time.
- Space: O(n * L) - storing all words across the groups plus keys.

The canonical key collapses all anagrams of a word to the same bucket. Using a 26-element count tuple instead of `sorted(word)` removes the O(L log L) sort, which matters for long strings. For Unicode input, a `frozenset(Counter(word).items())` or sorted key generalizes the approach.""",
    },
    {
        "prompt": "Given a non-negative integer `x`, compute and return the integer square root (floor of the true square root) without using any built-in sqrt function.",
        "answer": """<thinking>
I want the largest integer r with r*r <= x. Binary search over [0, x] for that r. For each mid, compare mid*mid to x and narrow the range. Track the best valid candidate.

Using a closed interval and comparing mid*mid <= x lets me move the lower bound up while remembering the candidate. I avoid overflow concerns since Python ints are arbitrary precision; in fixed-width languages I'd compare mid <= x // mid instead.
</thinking>

```python
def my_sqrt(x: int) -> int:
    if x < 2:
        return x
    lo, hi, ans = 1, x // 2, 0
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if mid * mid <= x:
            ans = mid       # candidate; try to go larger
            lo = mid + 1
        else:
            hi = mid - 1
    return ans
```

**Complexity and Explanation**
- Time: O(log x) - binary search over the answer space.
- Space: O(1) - constant extra variables.

We binary-search the monotonic predicate "mid*mid <= x", which is True up to the answer and False after, making it ideal for binary search. We store the last True `mid` as the answer. The bound `hi = x // 2` is a safe starting point since for x >= 2 the root never exceeds x/2. Newton's method converges even faster (quadratically) but binary search is simpler to get correct.""",
    },
    {
        "prompt": "Given the heights of bars in a histogram (array of non-negative integers, each bar width 1), compute how much rainwater can be trapped after raining.",
        "answer": """<thinking>
Water above each bar is bounded by min(tallest bar to its left, tallest bar to its right) minus its own height. The brute force computes left/right maxes per index in O(n^2).

The optimal O(n)/O(1) approach uses two pointers. The shorter side determines the water level, so I advance the pointer on the lower side, tracking running left_max and right_max. At each step the water trapped on the moving side is the local max minus the current height.
</thinking>

```python
from typing import List


def trap(height: List[int]) -> int:
    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0
    while left < right:
        if height[left] < height[right]:
            left_max = max(left_max, height[left])
            water += left_max - height[left]
            left += 1
        else:
            right_max = max(right_max, height[right])
            water += right_max - height[right]
            right -= 1
    return water
```

**Complexity and Explanation**
- Time: O(n) - each pointer moves inward at most n times total.
- Space: O(1) - four scalar variables.

The key insight: when `height[left] < height[right]`, the left bar's trapped water is fully determined by `left_max` alone, because we know some taller bar exists on the right. So we can safely commit the left side and advance. This avoids the O(n) extra arrays the prefix/suffix-max version needs while keeping linear time.""",
    },
    {
        "prompt": "Given an array `nums` and an integer `k`, return the k most frequent elements. You may return the answer in any order, and assume the answer is unique.",
        "answer": """<thinking>
First count frequencies in O(n). Then I need the top-k by frequency. A heap of size k gives O(n log k). But there's an even better O(n) approach: bucket sort by frequency, since frequencies range from 1 to n.

I'll use bucket sort: index buckets by frequency, place each element in its frequency bucket, then walk buckets from high frequency to low, collecting until I have k elements.
</thinking>

```python
from collections import Counter
from typing import List


def top_k_frequent(nums: List[int], k: int) -> List[int]:
    counts = Counter(nums)
    n = len(nums)
    buckets: List[List[int]] = [[] for _ in range(n + 1)]
    for value, freq in counts.items():
        buckets[freq].append(value)

    result: List[int] = []
    for freq in range(n, 0, -1):
        for value in buckets[freq]:
            result.append(value)
            if len(result) == k:
                return result
    return result
```

**Complexity and Explanation**
- Time: O(n) - counting is O(n), bucket fill is O(distinct), and the bucket scan is O(n) total.
- Space: O(n) - the counts map and buckets.

Because any frequency is at most n, we can index buckets directly by frequency rather than sorting them, giving linear time and beating the O(n log k) heap approach. Scanning buckets from highest frequency down and stopping at k yields exactly the most frequent elements. The heap method is preferable only when k is tiny and memory is tight.""",
    },
]


if __name__ == "__main__":
    assert len(examples) == 25, f"expected 25 examples, got {len(examples)}"
    for i, ex in enumerate(examples):
        assert "prompt" in ex and ex["prompt"].strip(), f"example {i} missing prompt"
        assert "answer" in ex and "<thinking>" in ex["answer"], f"example {i} malformed answer"
        assert "```python" in ex["answer"], f"example {i} missing code block"
        assert "Complexity" in ex["answer"], f"example {i} missing complexity analysis"
    print(f"All {len(examples)} examples validated.")
