# NeetCode 150 – Solutions (Python)

Solutions for the most commonly asked problems. Use alongside the worksheet for quick reference.

---

## Arrays & Hashing

### 1. Contains Duplicate (LC 217)
```python
def containsDuplicate(nums: list[int]) -> bool:
    seen = set()
    for n in nums:
        if n in seen:
            return True
        seen.add(n)
    return False
```
**O(n) time, O(n) space**

---

### 2. Valid Anagram (LC 242)
```python
def isAnagram(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    count = [0] * 26
    for c in s:
        count[ord(c) - ord('a')] += 1
    for c in t:
        count[ord(c) - ord('a')] -= 1
        if count[ord(c) - ord('a')] < 0:
            return False
    return True
```
**O(n) time, O(1) space (26 chars)**

---

### 3. Two Sum (LC 1)
```python
def twoSum(nums: list[int], target: int) -> list[int]:
    seen = {}
    for i, n in enumerate(nums):
        comp = target - n
        if comp in seen:
            return [seen[comp], i]
        seen[n] = i
    return []
```
**O(n) time, O(n) space**

---

### 4. Group Anagrams (LC 49)
```python
def groupAnagrams(strs: list[str]) -> list[list[str]]:
    from collections import defaultdict
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))
        groups[key].append(s)
    return list(groups.values())
```
**O(n·k log k) time, O(n·k) space**

---

### 5. Top K Frequent Elements (LC 347)
```python
def topKFrequent(nums: list[int], k: int) -> list[int]:
    from collections import Counter
    count = Counter(nums)
    bucket = [[] for _ in range(len(nums) + 1)]
    for num, freq in count.items():
        bucket[freq].append(num)
    res = []
    for i in range(len(bucket) - 1, -1, -1):
        res.extend(bucket[i])
        if len(res) >= k:
            return res[:k]
    return res[:k]
```
**O(n) time, O(n) space (bucket sort)**

---

### 6. Product of Array Except Self (LC 238)
```python
def productExceptSelf(nums: list[int]) -> list[int]:
    n = len(nums)
    res = [1] * n
    prefix = 1
    for i in range(n):
        res[i] = prefix
        prefix *= nums[i]
    suffix = 1
    for i in range(n - 1, -1, -1):
        res[i] *= suffix
        suffix *= nums[i]
    return res
```
**O(n) time, O(1) space (output不计)**

---

### 7. Longest Consecutive Sequence (LC 128)
```python
def longestConsecutive(nums: list[int]) -> int:
    s = set(nums)
    best = 0
    for n in s:
        if n - 1 not in s:  # start of streak
            length = 0
            while n + length in s:
                length += 1
            best = max(best, length)
    return best
```
**O(n) time, O(n) space**

---

## Two Pointers

### 8. Valid Palindrome (LC 125)
```python
def isPalindrome(s: str) -> bool:
    l, r = 0, len(s) - 1
    while l < r:
        while l < r and not s[l].isalnum():
            l += 1
        while l < r and not s[r].isalnum():
            r -= 1
        if s[l].lower() != s[r].lower():
            return False
        l, r = l + 1, r - 1
    return True
```
**O(n) time, O(1) space**

---

### 9. 3Sum (LC 15)
```python
def threeSum(nums: list[int]) -> list[list[int]]:
    nums.sort()
    res = []
    for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        l, r = i + 1, len(nums) - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s == 0:
                res.append([nums[i], nums[l], nums[r]])
                l += 1
                while l < r and nums[l] == nums[l - 1]:
                    l += 1
            elif s < 0:
                l += 1
            else:
                r -= 1
    return res
```
**O(n²) time, O(1) space**

---

### 10. Container With Most Water (LC 11)
```python
def maxArea(height: list[int]) -> int:
    l, r = 0, len(height) - 1
    best = 0
    while l < r:
        area = min(height[l], height[r]) * (r - l)
        best = max(best, area)
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1
    return best
```
**O(n) time, O(1) space**

---

### 11. Trapping Rain Water (LC 42)
```python
def trap(height: list[int]) -> int:
    l, r = 0, len(height) - 1
    maxL, maxR = height[l], height[r]
    water = 0
    while l < r:
        if maxL <= maxR:
            l += 1
            maxL = max(maxL, height[l])
            water += maxL - height[l]
        else:
            r -= 1
            maxR = max(maxR, height[r])
            water += maxR - height[r]
    return water
```
**O(n) time, O(1) space**

---

## Sliding Window

### 12. Longest Substring Without Repeating Characters (LC 3)
```python
def lengthOfLongestSubstring(s: str) -> int:
    seen = set()
    l = 0
    best = 0
    for r in range(len(s)):
        while s[r] in seen:
            seen.remove(s[l])
            l += 1
        seen.add(s[r])
        best = max(best, r - l + 1)
    return best
```
**O(n) time, O(min(n, 26)) space**

---

### 13. Longest Repeating Character Replacement (LC 424)
```python
def characterReplacement(s: str, k: int) -> int:
    count = {}
    l = 0
    maxF = 0
    for r in range(len(s)):
        count[s[r]] = count.get(s[r], 0) + 1
        maxF = max(maxF, count[s[r]])
        if (r - l + 1) - maxF > k:
            count[s[l]] -= 1
            l += 1
    return len(s) - l
```
**O(n) time, O(26) space**

---

### 14. Minimum Window Substring (LC 76)
```python
def minWindow(s: str, t: str) -> str:
    from collections import Counter
    need = Counter(t)
    have = 0
    need_count = len(need)
    res, res_len = "", float('inf')
    l = 0
    for r in range(len(s)):
        c = s[r]
        need[c] = need.get(c, 0) - 1
        if need[c] == 0:
            have += 1
        while have == need_count:
            if r - l + 1 < res_len:
                res_len = r - l + 1
                res = s[l:r+1]
            need[s[l]] += 1
            if need[s[l]] > 0:
                have -= 1
            l += 1
    return res
```
**O(n + m) time, O(m) space**

---

## Stack

### 15. Valid Parentheses (LC 20)
```python
def isValid(s: str) -> bool:
    stack = []
    pair = {')': '(', ']': '[', '}': '{'}
    for c in s:
        if c in pair:
            if not stack or stack[-1] != pair[c]:
                return False
            stack.pop()
        else:
            stack.append(c)
    return len(stack) == 0
```
**O(n) time, O(n) space**

---

### 16. Min Stack (LC 155)
```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self) -> None:
        if self.stack.pop() == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]
```
**O(1) all ops, O(n) space**

---

### 17. Daily Temperatures (LC 739)
```python
def dailyTemperatures(temps: list[int]) -> list[int]:
    stack = []
    res = [0] * len(temps)
    for i, t in enumerate(temps):
        while stack and t > temps[stack[-1]]:
            j = stack.pop()
            res[j] = i - j
        stack.append(i)
    return res
```
**O(n) time, O(n) space**

---

### 18. Largest Rectangle in Histogram (LC 84)
```python
def largestRectangleArea(heights: list[int]) -> int:
    stack = []
    heights.append(0)
    best = 0
    for i, h in enumerate(heights):
        while stack and h < heights[stack[-1]]:
            idx = stack.pop()
            width = i - stack[-1] - 1 if stack else i
            best = max(best, heights[idx] * width)
        stack.append(i)
    heights.pop()
    return best
```
**O(n) time, O(n) space**

---

## Binary Search

### 19. Binary Search (LC 704)
```python
def search(nums: list[int], target: int) -> int:
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = (l + r) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            l = mid + 1
        else:
            r = mid - 1
    return -1
```
**O(log n) time, O(1) space**

---

### 20. Search in Rotated Sorted Array (LC 33)
```python
def search(nums: list[int], target: int) -> int:
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = (l + r) // 2
        if nums[mid] == target:
            return mid
        if nums[l] <= nums[mid]:
            if nums[l] <= target < nums[mid]:
                r = mid - 1
            else:
                l = mid + 1
        else:
            if nums[mid] < target <= nums[r]:
                l = mid + 1
            else:
                r = mid - 1
    return -1
```
**O(log n) time, O(1) space**

---

### 21. Koko Eating Bananas (LC 875)
```python
import math

def minEatingSpeed(piles: list[int], h: int) -> int:
    lo, hi = 1, max(piles)
    while lo < hi:
        mid = (lo + hi) // 2
        hours = sum(math.ceil(p / mid) for p in piles)
        if hours <= h:
            hi = mid
        else:
            lo = mid + 1
    return lo
```
**O(n log max) time, O(1) space**

---

## Linked List

### 22. Reverse Linked List (LC 206)
```python
def reverseList(head):
    prev = None
    while head:
        nxt = head.next
        head.next = prev
        prev = head
        head = nxt
    return prev
```
**O(n) time, O(1) space**

---

### 23. Linked List Cycle (LC 141)
```python
def hasCycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```
**O(n) time, O(1) space**

---

### 24. Merge Two Sorted Lists (LC 21)
```python
def mergeTwoLists(l1, l2):
    dummy = cur = ListNode()
    while l1 and l2:
        if l1.val <= l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 or l2
    return dummy.next
```
**O(n + m) time, O(1) space**

---

## Trees

### 25. Invert Binary Tree (LC 226)
```python
def invertTree(root):
    if not root:
        return None
    root.left, root.right = root.right, root.left
    invertTree(root.left)
    invertTree(root.right)
    return root
```
**O(n) time, O(h) space**

---

### 26. Maximum Depth (LC 104)
```python
def maxDepth(root):
    if not root:
        return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))
```
**O(n) time, O(h) space**

---

### 27. Validate BST (LC 98)
```python
def isValidBST(root, lo=float('-inf'), hi=float('inf')):
    if not root:
        return True
    if not (lo < root.val < hi):
        return False
    return (isValidBST(root.left, lo, root.val) and
            isValidBST(root.right, root.val, hi))
```
**O(n) time, O(h) space**

---

### 28. Binary Tree Level Order Traversal (LC 102)
```python
from collections import deque

def levelOrder(root):
    if not root:
        return []
    q = deque([root])
    res = []
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        res.append(level)
    return res
```
**O(n) time, O(w) space**

---

## Dynamic Programming

### 29. Climbing Stairs (LC 70)
```python
def climbStairs(n: int) -> int:
    if n <= 2:
        return n
    prev2, prev1 = 1, 2
    for _ in range(3, n + 1):
        prev2, prev1 = prev1, prev2 + prev1
    return prev1
```
**O(n) time, O(1) space**

---

### 30. House Robber (LC 198)
```python
def rob(nums: list[int]) -> int:
    rob, not_rob = 0, 0
    for n in nums:
        rob, not_rob = not_rob + n, max(rob, not_rob)
    return max(rob, not_rob)
```
**O(n) time, O(1) space**

---

### 31. Coin Change (LC 322)
```python
def coinChange(coins: list[int], amount: int) -> int:
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for a in range(1, amount + 1):
        for c in coins:
            if a >= c:
                dp[a] = min(dp[a], dp[a - c] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
```
**O(amount × len(coins)) time, O(amount) space**

---

### 32. Maximum Subarray / Kadane (LC 53)
```python
def maxSubArray(nums: list[int]) -> int:
    cur = best = nums[0]
    for n in nums[1:]:
        cur = max(n, cur + n)
        best = max(best, cur)
    return best
```
**O(n) time, O(1) space**

---

## Graphs

### 33. Number of Islands (LC 200)
```python
def numIslands(grid: list[list[str]]) -> int:
    m, n = len(grid), len(grid[0])

    def dfs(i, j):
        if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] == '0':
            return
        grid[i][j] = '0'
        for di, dj in [(0,1),(1,0),(0,-1),(-1,0)]:
            dfs(i + di, j + dj)

    count = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                count += 1
                dfs(i, j)
    return count
```
**O(m·n) time, O(m·n) space (recursion)**

---

### 34. Course Schedule (LC 207)
```python
def canFinish(numCourses: int, prerequisites: list[list[int]]) -> bool:
    from collections import defaultdict
    adj = defaultdict(list)
    for a, b in prerequisites:
        adj[a].append(b)
    state = [0] * numCourses  # 0 unvisited, 1 visiting, 2 done

    def dfs(u):
        if state[u] == 1:
            return False
        if state[u] == 2:
            return True
        state[u] = 1
        for v in adj[u]:
            if not dfs(v):
                return False
        state[u] = 2
        return True

    for i in range(numCourses):
        if not dfs(i):
            return False
    return True
```
**O(V + E) time, O(V) space**

---

## Bit Manipulation

### 35. Single Number (LC 136)
```python
def singleNumber(nums: list[int]) -> int:
    res = 0
    for n in nums:
        res ^= n
    return res
```
**O(n) time, O(1) space**

---

### 36. Number of 1 Bits (LC 191)
```python
def hammingWeight(n: int) -> int:
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count
```
**O(1) time (32 bits), O(1) space**

---

*For full NeetCode 150 solutions, visit [NeetCode.io](https://neetcode.io) or [LeetCode](https://leetcode.com).*
