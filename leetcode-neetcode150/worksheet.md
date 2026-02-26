# NeetCode 150 – 5-Minute Discussion Worksheet

Use this worksheet for a **5–10 minute warm-up** per problem. Fill in the 5-line algorithm summary and the complexity line. Ideal for rapid recall before coding.

---

## How to Use

1. **Line 1–5:** Core algorithm in 5 lines (problem → approach → key steps → edge cases → output).
2. **Complexity:** Time O(?), Space O(?) in one line.
3. **Discussion tip:** Say it out loud in under 2 minutes.

---

## Arrays & Hashing

### 1. Contains Duplicate
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Use hash set. (2) Iterate nums. (3) If num in set → return True. (4) Else add num to set. (5) Return False. **O(n), O(n)**

---

### 2. Valid Anagram
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) If len(s) ≠ len(t) return False. (2) Use char count dict (or array[26]). (3) Increment for s, decrement for t. (4) All counts should be 0. (5) Alternative: sort both, compare. **O(n), O(1)** (26 chars)

---

### 3. Two Sum
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Hash map: value → index. (2) For i, num in enumerate(nums): complement = target - num. (3) If complement in map → return [map[complement], i]. (4) Else map[num] = i. (5) One pass. **O(n), O(n)**

---

### 4. Group Anagrams
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) HashMap: key = sorted str or tuple(count). (2) For each str, key = ''.join(sorted(str)). (3) Append str to map[key]. (4) Return list(map.values()). (5) Anagrams share same key. **O(n·k log k), O(n·k)** (k = max str len)

---

### 5. Top K Frequent Elements
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Count freq with Counter. (2) Bucket sort: bucket[i] = list of nums with freq i. (3) Or min-heap of size k: push (-freq, num), pop if len>k. (4) Bucket: iterate from high freq to low, collect k elements. (5) Return top k. **O(n) bucket, O(n log k) heap**

---

### 6. Product of Array Except Self
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) answer[i] = prefix[i] * suffix[i]. (2) One pass left: prefix[i] = prefix[i-1] * nums[i-1]. (3) One pass right: suffix. (4) Or: single pass, maintain running prefix, then reverse pass for suffix. (5) O(1) space: use output array for prefix, then multiply suffix in reverse. **O(n), O(1)** (output不计)

---

### 7. Valid Sudoku
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) 9 sets each: rows, cols, boxes. (2) box_id = (r//3)*3 + c//3. (3) For each cell, if '.' skip. (4) If num in row/col/box set → return False. (5) Add to all three sets. **O(81), O(1)**

---

### 8. Longest Consecutive Sequence
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Convert nums to set. (2) For each num, only start streak if num-1 not in set (avoid recounting). (3) While num+length in set: length++. (4) Update max_len. (5) Each element visited at most twice. **O(n), O(n)**

---

## Two Pointers

### 9. Valid Palindrome
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Two pointers L=0, R=len-1. (2) While L<R: skip non-alphanumeric. (3) Compare s[L].lower() vs s[R].lower(). (4) If != return False. (5) L++, R--. **O(n), O(1)**

---

### 10. Two Sum II (Sorted)
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) L=0, R=len-1. (2) Sum = numbers[L]+numbers[R]. (3) If sum==target return [L+1,R+1]. (4) If sum<target L++; else R--. (5) Guaranteed one solution. **O(n), O(1)**

---

### 11. 3Sum
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Sort array. (2) For i, fix nums[i], two pointers L=i+1, R=n-1. (3) Sum = nums[i]+nums[L]+nums[R]. (4) If sum==0: append, skip duplicates for L/R. (5) If sum<0 L++; else R--. **O(n²), O(1)** (sort in-place)

---

### 12. Container With Most Water
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) L=0, R=len-1. (2) Area = min(h[L],h[R]) * (R-L). (3) Move pointer at smaller height inward (greedy: we want higher walls). (4) Update max_area. (5) Stop when L>=R. **O(n), O(1)**

---

### 13. Trapping Rain Water
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Two pointers L, R; maxL, maxR. (2) Water at i = min(maxL,maxR) - h[i]. (3) If h[L]<=h[R]: water depends on maxL, L++. (4) Else: depends on maxR, R--. (5) Greedy: always fill from shorter side. **O(n), O(1)**

---

## Sliding Window

### 14. Best Time to Buy and Sell Stock
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) One pass: keep min_price, max_profit. (2) For price: profit = price - min_price. (3) max_profit = max(max_profit, profit). (4) min_price = min(min_price, price). (5) Return max_profit. **O(n), O(1)**

---

### 15. Longest Substring Without Repeating Characters
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Sliding window + set. (2) R expands: add s[R] to set. (3) If duplicate: L shrinks until s[R] not in set. (4) Remove s[L], L++. (5) ans = max(ans, R-L+1). **O(n), O(min(n,26))**

---

### 16. Longest Repeating Character Replacement
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Sliding window. (2) Count freq of chars in window. (3) maxFreq = max(count.values()). (4) If window_len - maxFreq > k: invalid, shrink L. (5) ans = max(ans, R-L+1). Key: (len - maxFreq) ≤ k means replaceable. **O(n), O(26)**

---

### 17. Minimum Window Substring
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Need dict for t, have dict for window. (2) Expand R: add s[R], if have==need: try shrink. (3) Shrink L: while have==need, remove s[L], L++. (4) Update min_len and result. (5) Need: all chars in t with correct counts. **O(n+m), O(m)**

---

### 18. Sliding Window Maximum
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Monotonic decreasing deque: store indices. (2) Front = max of current window. (3) When R moves: pop from back while nums[back] < nums[R]. (4) Append R. (5) When L passes deque front: popleft. **O(n), O(k)**

---

## Stack

### 19. Valid Parentheses
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Stack. Map )→(, ]→[, }→{. (2) For c: if open, push. (3) If close: if stack empty or stack.pop()≠map[c] return False. (4) End: return len(stack)==0. (5) Odd len → False. **O(n), O(n)**

---

### 20. Min Stack
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Two stacks: main + min_stack. (2) Push: main.push(x). (3) If min_stack empty or x<=min_stack[-1]: min_stack.push(x). (4) Pop: if main.pop()==min_stack[-1]: min_stack.pop(). (5) getMin: min_stack[-1]. **O(1) all ops, O(n)**

---

### 21. Evaluate Reverse Polish Notation
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Stack. (2) For token: if digit, push. (3) If op: pop b, pop a, compute a op b, push. (4) Order: second pop is left operand. (5) Return stack[0]. **O(n), O(n)**

---

### 22. Generate Parentheses
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Backtrack: (open, close). (2) Base: len(cur)==2n → append. (3) If open<n: add '(', recurse. (4) If close<open: add ')', recurse. (5) Only valid: close never exceeds open. **O(4^n/√n), O(n)** recursion

---

### 23. Daily Temperatures
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Monotonic decreasing stack (indices). (2) For i, t: while stack and t > temps[stack[-1]]: (3) j = stack.pop(), ans[j] = i - j. (4) stack.append(i). (5) Remaining in stack: 0 (default). **O(n), O(n)**

---

### 24. Largest Rectangle in Histogram
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Monotonic stack (indices). (2) For i, h: while stack and h < heights[stack[-1]]: (3) pop idx, height = heights[idx], width = i - stack[-1] - 1 (or i if stack empty). (4) area = height * width. (5) Append i. Pad heights with 0 at end. **O(n), O(n)**

---

## Binary Search

### 25. Binary Search
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) L=0, R=len-1. (2) While L<=R: mid=(L+R)//2. (3) If nums[mid]==target return mid. (4) If nums[mid]<target: L=mid+1. (5) Else R=mid-1. Return -1. **O(log n), O(1)**

---

### 26. Search in Rotated Sorted Array
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) L, R. mid. (2) If nums[mid]==target return. (3) If left half sorted (nums[L]<=nums[mid]): if target in [L,mid] R=mid-1 else L=mid+1. (4) Else right sorted: if target in [mid,R] L=mid+1 else R=mid-1. (5) One half always sorted. **O(log n), O(1)**

---

### 27. Find Minimum in Rotated Sorted Array
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) L, R. (2) If nums[L]<=nums[R]: already sorted, return nums[L]. (3) Mid. (4) If nums[mid]>=nums[L]: min in right, L=mid+1. (5) Else: min in left incl mid, R=mid. **O(log n), O(1)**

---

### 28. Koko Eating Bananas
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Binary search on k (speed): lo=1, hi=max(piles). (2) For mid: hours = sum(ceil(p/mid) for p in piles). (3) If hours<=h: try smaller k, hi=mid. (4) Else: need faster, lo=mid+1. (5) Return lo. **O(n log max), O(1)**

---

## Linked List

### 29. Reverse Linked List
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) prev=None. (2) While head: next_node=head.next. (3) head.next=prev. (4) prev=head, head=next_node. (5) Return prev. **O(n), O(1)**

---

### 30. Merge Two Sorted Lists
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Dummy node. cur=dummy. (2) While l1 and l2: if l1.val<=l2.val: cur.next=l1, l1=l1.next. (3) Else: cur.next=l2, l2=l2.next. (4) cur=cur.next. (5) cur.next=l1 or l2. Return dummy.next. **O(n+m), O(1)**

---

### 31. Linked List Cycle
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Floyd: slow, fast = head, head. (2) While fast and fast.next: slow=slow.next, fast=fast.next.next. (3) If slow==fast: cycle exists. (4) To find start: reset slow=head, move both 1 step until meet. (5) Meet point = cycle start. **O(n), O(1)**

---

### 32. LRU Cache
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) OrderedDict or DLL + HashMap. (2) get: if key in map, move to end (most recent), return value. (3) put: if key exists, update value, move to end. (4) Else: add to end; if len>capacity, remove front (least recent). (5) HashMap: key→node for O(1) access. **O(1) get/put, O(capacity)**

---

## Trees

### 33. Invert Binary Tree
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Base: root is None return None. (2) Swap root.left, root.right. (3) invert(root.left). (4) invert(root.right). (5) Return root. **O(n), O(h)** recursion

---

### 34. Maximum Depth of Binary Tree
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Base: root None return 0. (2) left = maxDepth(root.left). (3) right = maxDepth(root.right). (4) return 1 + max(left, right). (5) BFS: count levels. **O(n), O(h)**

---

### 35. Diameter of Binary Tree
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Diameter = max path between any two nodes (may not pass root). (2) For each node: path thru node = 1 + left_height + right_height. (3) DFS returns height. (4) Global max_diameter = max(path_thru_each_node). (5) height = 1 + max(L,R). **O(n), O(h)**

---

### 36. Validate BST
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) DFS(root, lo=-inf, hi=inf). (2) If root None return True. (3) If not (lo < root.val < hi): return False. (4) return dfs(left, lo, root.val) and dfs(right, root.val, hi). (5) Pass valid range down. **O(n), O(h)**

---

### 37. Binary Tree Level Order Traversal
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) BFS with queue. (2) While queue: level_size=len(queue). (3) level=[]; for _ in range(level_size): node=queue.popleft(), level.append(node.val). (4) Add children to queue. (5) result.append(level). **O(n), O(w)** w=width

---

### 38. Binary Tree Maximum Path Sum
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) DFS returns max path sum from node (down one branch). (2) left = max(0, dfs(left)); right = max(0, dfs(right)). (3) Path thru node = root.val + left + right. (4) Update global max_path. (5) Return root.val + max(left,right) (single branch). **O(n), O(h)**

---

## Heap

### 39. Kth Largest Element in Array
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Min-heap of size k. (2) For num: push num; if len>k pop (smallest). (3) Top of min-heap = kth largest. (4) Or QuickSelect: O(n) avg. (5) Heap: O(n log k). **O(n log k) heap, O(n) QuickSelect**

---

### 40. Merge K Sorted Lists
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Min-heap: (val, list_idx, node). (2) Push (head.val, i, head) for each list. (3) While heap: pop smallest, append to result, push next from that list. (4) Dummy + curr. (5) N total nodes. **O(N log k)** k lists

---

### 41. Find Median from Data Stream
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Max-heap (left half), min-heap (right half). (2) Max-heap stores negatives for max. (3) addNum: push to one heap, balance so len_diff ≤ 1. (4) findMedian: if equal size (max_top+min_top)/2; else top of larger. (5) Balance: pop from one, push to other. **O(log n) add, O(1) median, O(n)**

---

## Backtracking

### 42. Subsets
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Backtrack(i, path). (2) Append path copy to result (each path is valid). (3) For j in range(i, n): path.append(nums[j]), backtrack(j+1), path.pop(). (4) Include/exclude each element. (5) 2^n subsets. **O(n·2^n), O(n)**

---

### 43. Combination Sum
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Backtrack(i, path, total). (2) If total==target: append path, return. (3) If total>target or i>=n: return. (4) Include: path.append(c[i]), backtrack(i, path, total+c[i]), path.pop(). (5) Exclude: backtrack(i+1, path, total). Reuse allowed. **O(2^target), O(target)**

---

### 44. Permutations
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Backtrack(path). (2) If len(path)==n: append. (3) For num in nums: if num not in path: path.append(num), backtrack(path), path.pop(). (4) Use used[] to avoid in-path check. (5) n! permutations. **O(n·n!), O(n)**

---

### 45. Word Search
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) DFS from each cell. (2) dfs(r,c,i): if i==len(word) return True. (3) If out of bounds or board[r][c]!=word[i]: return False. (4) Mark visited, recurse 4 dirs. (5) Unmark (backtrack). **O(m·n·4^L), O(L)** L=word len

---

## Graphs

### 46. Number of Islands
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) For each cell: if '1', count++, DFS/BFS to mark all connected '1' as visited. (2) DFS: out of bounds or water → return. (3) Mark (r,c) visited (or flip to '0'). (4) Recurse 4 dirs. (5) Each cell visited once. **O(m·n), O(m·n)** worst recursion

---

### 47. Clone Graph
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) HashMap: old_node → new_node. (2) DFS(node): if node in map return map[node]. (3) Create copy, map[node]=copy. (4) For neighbor: copy.neighbors.append(dfs(neighbor)). (5) Return copy. **O(V+E), O(V)**

---

### 48. Course Schedule (Cycle Detection)
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Build adj list. (2) Three states: 0=unvisited, 1=visiting, 2=done. (3) DFS: if state==1 cycle; if 2 return True. (4) Mark 1, recurse neighbors, mark 2. (5) Return True if no cycle. **O(V+E), O(V)**

---

### 49. Word Ladder
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) BFS from beginWord. (2) Queue: (word, steps). (3) For each word, try changing each char to a-z. (4) If new word in wordList: add to queue, remove from wordList. (5) If new word==endWord return steps+1. **O(M²·N)** M=word len, N=wordList

---

## Dynamic Programming (1-D)

### 50. Climbing Stairs
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) dp[i] = ways to reach i. (2) dp[0]=1, dp[1]=1. (3) dp[i] = dp[i-1] + dp[i-2]. (4) Fibonacci. (5) O(1) space: prev2, prev1, curr. **O(n), O(1)**

---

### 51. House Robber
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) dp[i] = max money robbing first i houses. (2) dp[i] = max(dp[i-1], nums[i-1]+dp[i-2]). (3) Rob i: nums[i]+dp[i-2]; skip: dp[i-1]. (4) Base: dp[0]=0, dp[1]=nums[0]. (5) O(1) space: rob, not_rob. **O(n), O(1)**

---

### 52. Coin Change
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) dp[amount] = min coins. (2) dp[0]=0. (3) For a 1..amount: dp[a]=min(dp[a-c]+1 for c in coins if a>=c). (4) If no valid: dp[a]=inf. (5) Return dp[amount] if < inf else -1. **O(amount·len(coins)), O(amount)**

---

### 53. Longest Increasing Subsequence
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) dp[i] = LIS ending at i. (2) dp[i]=1+max(dp[j] for j<i if nums[j]<nums[i]). (3) Return max(dp). (4) O(n²). (5) Patience: binary search, O(n log n). **O(n²) or O(n log n)**

---

### 54. Word Break
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) dp[i] = s[:i] can be segmented. (2) dp[0]=True. (3) For i 1..n: dp[i]=any(dp[j] and s[j:i] in wordDict for j<i). (4) wordSet for O(1) lookup. (5) Return dp[n]. **O(n²·m)** m=avg word len

---

## Greedy

### 55. Maximum Subarray (Kadane)
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) cur_sum, max_sum. (2) For num: cur_sum = max(num, cur_sum+num). (3) max_sum = max(max_sum, cur_sum). (4) cur_sum = local max ending here. (5) One pass. **O(n), O(1)**

---

### 56. Jump Game
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) max_reach = 0. (2) For i, jump: if i > max_reach return False. (3) max_reach = max(max_reach, i+jump). (4) If max_reach >= n-1 return True. (5) Greedy: extend reach each step. **O(n), O(1)**

---

### 57. Merge Intervals
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Sort by start. (2) res=[intervals[0]]. (3) For int in intervals[1:]: if int[0]<=res[-1][1]: merge, res[-1][1]=max(res[-1][1],int[1]). (4) Else: append. (5) Overlapping: next.start <= last.end. **O(n log n), O(n)**

---

## Bit Manipulation

### 58. Single Number
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) XOR: a^a=0, a^0=a. (2) XOR all numbers. (3) Pairs cancel out. (4) Result = single number. (5) return reduce(xor, nums). **O(n), O(1)**

---

### 59. Number of 1 Bits
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) Count set bits. (2) While n: count += n&1, n>>=1. (3) Or: while n: n &= n-1 (clear lowest 1), count++. (4) n & (n-1) removes rightmost 1. (5) O(1) ops per bit. **O(1)** (32 bits)

---

### 60. Counting Bits
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer:** (1) ans[0]=0. (2) ans[i] = ans[i>>1] + (i&1). (3) i has same bits as i/2 plus LSB. (4) Or: ans[i] = ans[i & (i-1)] + 1. (5) DP: build from smaller. **O(n), O(1)** output
