# NeetCode 150 – 5-Minute Discussion Worksheet

Use this worksheet for a **5–10 minute warm-up** per problem. Fill in the 5-line algorithm summary and the complexity line. Ideal for rapid recall before coding.

---

## How to Use

1. **Line 1–5:** Core algorithm in 5 lines (problem → approach → key steps → edge cases → output).
2. **Complexity:** Time O(?), Space O(?) in one line.
3. **Answer (discussion):** Use the answer block to rehearse interview talk: *Choices* (what approaches?), *Brute force* (naive + complexity), *Optimization* (key insight), *Trade-off* (time/space vs alternatives).
4. **Discussion tip:** Say it out loud in under 2 minutes.

---

## Arrays & Hashing

### 1. Contains Duplicate
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Sort then check adjacent pairs, or use a hash set. *Brute force:* Two nested loops — for each i, check if nums[i] appears in nums[i+1:]. O(n²) time, O(1) space. *Optimization:* Single pass with a set; if we've seen the value before, return true. *Trade-off:* We pay O(n) space for O(n) time. **O(n), O(n)**

---

### 2. Valid Anagram
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Sort both and compare; or count character frequencies. *Brute force:* For each char in s, find and remove one occurrence in t (or nested loop). O(n²) or O(n) with list removal. *Optimization:* One array of size 26: increment for s, decrement for t; if any count goes negative, not anagram. *Trade-off:* Sort is O(n log n) time, O(n) space; count is O(n) time, O(1) space (fixed 26). **O(n), O(1)**

---

### 3. Two Sum
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Brute force two loops; or one pass with a hash map (value → index). *Brute force:* For each i, for each j > i, if nums[i] + nums[j] == target return [i,j]. O(n²) time, O(1) space. *Optimization:* As we scan, store each number and its index. For current num, check if (target - num) is already in the map. One pass. *Trade-off:* O(n) space for O(n) time; can't do better than one pass without extra space. **O(n), O(n)**

---

### 4. Group Anagrams
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Key by sorted string; or key by tuple of character counts (no sort). *Brute force:* For each string, compare against all others to see if anagram (compare sorted or count chars). O(n²·k) with comparisons. *Optimization:* Group by a canonical key. Sorted string is simplest: anagrams share the same sorted form. Hash map key → list of strings. *Trade-off:* Sort key costs O(k log k) per string; count tuple is O(k) but more code. **O(n·k log k), O(n·k)**

---

### 5. Top K Frequent Elements
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Sort by frequency (O(n log n)); min-heap of size k (O(n log k)); or bucket sort by frequency (O(n)). *Brute force:* Count all frequencies, then sort (num, freq) by freq descending and take first k. O(n log n). *Optimization:* (1) Bucket sort: bucket[freq] = list of numbers; max freq ≤ n, so iterate buckets from high to low and collect k. (2) Min-heap of size k: push (freq, num), pop smallest when size > k; top k stay. *Trade-off:* Bucket is O(n) but O(n) space for buckets; heap is O(n log k) and O(k) space. **O(n) bucket, O(n log k) heap**

---

### 6. Product of Array Except Self
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Precompute prefix and suffix arrays; or use output array for prefix then overwrite with suffix in one more pass (O(1) extra space). *Brute force:* For each i, compute product of all j≠i in two loops. O(n²) time. *Optimization:* answer[i] = (product of nums[0..i-1]) × (product of nums[i+1..n-1]). Two passes: fill output with prefix products, then multiply by suffix from the right. *Trade-off:* If output doesn't count as extra space, we get O(n) time, O(1) extra space. **O(n), O(1)**

---

### 7. Valid Sudoku
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Check row/col/box with sets; or use fixed-size arrays (e.g. 9 booleans per row/col/box). *Brute force:* For each filled cell, scan its row, column, and 3×3 box for duplicates. Many repeated checks, messy. *Optimization:* One pass: for each (r,c) with digit d, maintain sets for rows[r], cols[c], boxes[box_id]. box_id = (r//3)*3 + c//3. If d already in any of the three, invalid. *Trade-off:* 9×9 grid → O(1) time and space; sets are clean, arrays would be slightly tighter. **O(81), O(1)**

---

### 8. Longest Consecutive Sequence
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Sort then scan for consecutive runs O(n log n); or set + "start of streak" idea for O(n). *Brute force:* Sort, then one pass counting current run. O(n log n) time, O(1) or O(n) space depending on sort. *Optimization:* Put all in a set. Only start expanding from n if (n-1) is not in the set — so each streak is counted once. For each start, extend while (n+length) in set. *Trade-off:* O(n) time with set; we need O(n) space. Sorting avoids extra space but is slower. **O(n), O(n)**

---

## Two Pointers

### 9. Valid Palindrome
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Two pointers from both ends; or build a filtered string (alphanumeric only) and check if it equals its reverse. *Brute force:* Build new string with only alphanumeric, lowercased; compare with reverse. O(n) time, O(n) space. *Optimization:* L=0, R=len-1. Skip non-alphanumeric; compare chars (ignore case); if mismatch return false. Move L and R inward. *Trade-off:* Two pointers give O(1) extra space; building a string is simpler but uses O(n) space. **O(n), O(1)**

---

### 10. Two Sum II (Sorted)
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Same as Two Sum — hash map one pass; or use two pointers (possible because array is sorted). *Brute force:* Two nested loops, or one pass with hash map. O(n²) or O(n) with map. *Optimization:* Sorted ⇒ if sum too small, increase L; if too big, decrease R. L=0, R=len-1; move one pointer per step. *Trade-off:* Two pointers O(n) time, O(1) space; no need for hash map when sorted. **O(n), O(1)**

---

### 11. 3Sum
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Hash map for each pair (like 2Sum); or sort + fix one number and two pointers for the other two. *Brute force:* Three nested loops, check if sum==0, dedupe triplets. O(n³). *Optimization:* Sort. For each i, two pointers L=i+1, R=n-1 to find pairs that sum to -nums[i]. Skip duplicate i and duplicate L/R. *Trade-off:* Sort gives structure; O(n²) time, O(1) extra space (sort in-place). Hash approach can be O(n²) but more space and dedup logic. **O(n²), O(1)**

---

### 12. Container With Most Water
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Try all pairs O(n²); or two pointers from both ends. *Brute force:* For every (i,j), area = min(h[i],h[j])*(j-i); return max. O(n²). *Optimization:* Start L=0, R=len-1. Area is limited by the shorter side. Moving the pointer at the shorter height inward might get a taller line; moving the taller one inward only shrinks width. So always move the shorter pointer. *Trade-off:* One pass O(n), O(1). Greedy works because we're optimizing over two dimensions (width and min height). **O(n), O(1)**

---

### 13. Trapping Rain Water
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* For each index compute water from left-max and right-max (two passes); or two pointers with maxL/maxR. *Brute force:* For each i, find max height to the left and right, add min(maxL,maxR)-height[i]. O(n²) if we scan each time; O(n) with two precomputed arrays. *Optimization:* Two pointers L, R and track maxL, maxR. Water at current pointer is determined by the smaller of maxL and maxR. Move the pointer that has the smaller max (we can't get more water on that side). *Trade-off:* Two-pointer gives O(n) time, O(1) space; precomputed arrays O(n) time, O(n) space. **O(n), O(1)**

---

## Sliding Window

### 14. Best Time to Buy and Sell Stock
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Two loops (buy day i, sell day j>i); or one pass tracking min so far and best profit. *Brute force:* For each i, for each j>i, profit = prices[j]-prices[i]; take max. O(n²). *Optimization:* One pass: maintain min_price seen so far. For each price, profit if we sold today = price - min_price; update max_profit. Then update min_price. We're effectively trying every sell day with the best buy day so far. *Trade-off:* O(n) time, O(1) space. **O(n), O(1)**

---

### 15. Longest Substring Without Repeating Characters
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* For each start index expand until duplicate (n starts, each can go to n); or sliding window with a set. *Brute force:* For each i, for each j>=i, check if s[i:j+1] has no repeats (e.g. with set). O(n²) or O(n²) with early exit. *Optimization:* Sliding window: expand R, add s[R] to set. If s[R] already in set, shrink L (remove s[L]) until window is valid again. Track max length. Each char enters and leaves at most once. *Trade-off:* O(n) time; space O(min(n, charset size)). **O(n), O(min(n,26))**

---

### 16. Longest Repeating Character Replacement
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Try each char as the "majority" and run sliding window; or one window with "replacements = len - maxFreq". *Brute force:* For each (L,R), count freq, compute replacements needed; if ≤k update ans. O(n²) or O(26·n) if we fix majority. *Optimization:* Sliding window. Valid window ⟺ (window_len - maxFreq) ≤ k (we can replace the rest to match majority). Expand R; when invalid, shrink L. maxFreq can be updated as we add/remove; we don't need to decrement maxFreq when we shrink (we only care about max in current window). *Trade-off:* O(n) time, O(26) space. **O(n), O(26)**

---

### 17. Minimum Window Substring
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Sliding window (expand/shrink); or for each start, scan until we have all of t. *Brute force:* For each i, for each j>=i, check if s[i:j+1] contains every char in t with enough counts. O(n²) or similar. *Optimization:* Sliding window. "have" = how many distinct chars in t are satisfied in current window. Expand R to add chars; when have==need, try shrinking L (remove from left) while window still valid. Track smallest valid window. *Trade-off:* O(n+m) time; we need counts for t and possibly window. **O(n+m), O(m)**

---

### 18. Sliding Window Maximum
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* For each window compute max (heap or linear scan); or monotonic deque. *Brute force:* For each window of size k, scan to find max. O(n·k). Or use a max-heap: push indices, pop when top is outside window. O(n log n). *Optimization:* Deque stores indices in decreasing order of value. Front = current window max. When adding R: pop from back while back index has value < nums[R] (they can never be max again). When L moves past front, popleft. *Trade-off:* Each index pushed and popped at most once ⇒ O(n) time, O(k) space. **O(n), O(k)**

---

## Stack

### 19. Valid Parentheses
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Stack to match closing with most recent open; or count balance (doesn't work for mixed types like ([)]. *Brute force:* Repeatedly find and remove matching pairs "()", "[]", "{}" until string empty or no change. O(n²). *Optimization:* One pass: open brackets push to stack; close bracket must match stack top and pop. Map close→open for lookup. Invalid if stack empty on close or wrong type, or non-empty at end. *Trade-off:* O(n) time, O(n) space; stack depth ≤ n/2. **O(n), O(n)**

---

### 20. Min Stack
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Store (value, min_so_far) in one stack; or separate min_stack that only pushes when we see a new min. *Brute force:* On getMin, scan all stored values. O(1) push/pop, O(n) getMin. *Optimization:* Second stack "min_stack": push to min_stack only when pushing a value ≤ current min (so min_stack is non-increasing). On pop, pop min_stack only if we're popping the current min. getMin = min_stack top. *Trade-off:* All ops O(1); extra space O(n) worst case (e.g. decreasing sequence). **O(1) all ops, O(n)**

---

### 21. Evaluate Reverse Polish Notation
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Stack (postfix); or recursion/expression tree. *Brute force:* Not really a "brute force" — RPN is designed for stack. Could parse and build tree then evaluate. *Optimization:* Left-to-right: numbers go on stack; operator pops two (right first, then left), applies op, pushes result. Order matters: first pop = right operand. *Trade-off:* One pass O(n), stack size O(n) for deeply nested expressions. **O(n), O(n)**

---

### 22. Generate Parentheses
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Backtrack (add '(' or ')' with constraints); or iterative with stack. *Brute force:* Generate all 2^(2n) sequences of n '(' and n ')', filter to valid. Very wasteful. *Optimization:* Backtrack: we need open count ≤ n and close count ≤ open. If open<n we can add '('; if close<open we can add ')'. Base: when len==2n, we have one valid string. *Trade-off:* Catalan number of solutions; time O(4^n/√n), recursion stack O(n). **O(4^n/√n), O(n)**

---

### 23. Daily Temperatures
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* For each day, scan right until warmer (O(n²)); or monotonic stack (indices of days waiting for a warmer). *Brute force:* For each i, ans[i] = smallest j>i with temps[j]>temps[i], else 0. O(n²). *Optimization:* Stack holds indices of days we haven't found a warmer day for. When we see a warmer temp at i, it resolves all stack tops that are colder; ans[pop()] = i - pop(). Then push i. *Trade-off:* Each index pushed and popped once ⇒ O(n) time, O(n) space. **O(n), O(n)**

---

### 24. Largest Rectangle in Histogram
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* For each bar, extend left/right until lower (O(n) per bar ⇒ O(n²)); or monotonic stack. *Brute force:* For each index i, find left and right boundaries (first smaller height), area = height[i] * (right-left-1). O(n²) or O(n) with two precomputed arrays (previous/next smaller). *Optimization:* Monotonic stack (indices, increasing heights). When we pop (height h), the bar that caused the pop is first right smaller; stack top is first left smaller. Width = (current_i - left - 1). Pad with 0 at end to pop remaining. *Trade-off:* One pass O(n), O(n) stack. **O(n), O(n)**

---

## Binary Search

### 25. Binary Search
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Linear scan O(n); or binary search on sorted array. *Brute force:* Scan left to right until we find target or pass it. O(n). *Optimization:* Sorted ⇒ binary search: mid = (L+R)//2; if nums[mid]==target return; if nums[mid]<target search right (L=mid+1); else search left (R=mid-1). *Trade-off:* O(log n) time, O(1) space. Standard pattern for "find in sorted". **O(log n), O(1)**

---

### 26. Search in Rotated Sorted Array
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Linear scan O(n); or modified binary search using "one half is always sorted". *Brute force:* Linear search. O(n). *Optimization:* At mid, one of [L,mid] or [mid,R] is sorted (no rotation in that half). If target lies in the sorted half, search there; else search the other half. Compare target with nums[L], nums[mid], nums[R] to decide. *Trade-off:* O(log n) time, O(1) space. Duplicates can break "one half sorted" (then use linear or shrink). **O(log n), O(1)**

---

### 27. Find Minimum in Rotated Sorted Array
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Linear scan for min O(n); or binary search — min is where the "break" is. *Brute force:* One pass, track minimum. O(n). *Optimization:* If nums[L]<=nums[R], segment is sorted, min=nums[L]. Else rotated: mid. If nums[mid]>=nums[L], left half is sorted so min is in right (L=mid+1); else min in left including mid (R=mid). *Trade-off:* O(log n) time, O(1) space. **O(log n), O(1)**

---

### 28. Koko Eating Bananas
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Binary search on speed k; or try k=1,2,... until hours<=h (slow). *Brute force:* For k=1 to max(piles), compute total hours; return smallest k with hours<=h. O(n · max). *Optimization:* Binary search k in [1, max(piles)]. For a given k, hours = sum(ceil(pile/k)). If hours<=h we can try smaller k (hi=mid); else need larger k (lo=mid+1). *Trade-off:* O(n log max) time, O(1) space. **O(n log max), O(1)**

---

## Linked List

### 29. Reverse Linked List
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Iterative (three pointers: prev, curr, next); or recursion (reverse rest, then point rest to curr). *Brute force:* Copy list to array, reverse array, rebuild list. O(n) time, O(n) space. *Optimization:* Iterative: prev=None. For each node, save next, set curr.next=prev, advance prev=curr and curr=next. *Trade-off:* Iterative O(n) time, O(1) space; recursion O(n) time, O(n) stack. **O(n), O(1)**

---

### 30. Merge Two Sorted Lists
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Merge with dummy node (in-place pointer wiring); or new list (extra space). *Brute force:* Collect all values, sort, build new list. O((n+m) log(n+m)) time. *Optimization:* Dummy head; cur = dummy. While both lists non-null, attach the smaller node to cur and advance. At end, attach remainder. *Trade-off:* O(n+m) time, O(1) space (we only rewire pointers). **O(n+m), O(1)**

---

### 31. Linked List Cycle
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Hash set of visited nodes; or Floyd's slow/fast pointers. *Brute force:* Traverse and put each node in a set; if we see a node already in set, cycle. O(n) time, O(n) space. *Optimization:* Floyd: slow moves 1, fast moves 2. If they meet, cycle exists. To find start: reset slow to head, move both 1 step until they meet; that node is cycle start (math: distance from head to start = distance from meet to start). *Trade-off:* O(n) time, O(1) space. **O(n), O(1)**

---

### 32. LRU Cache
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* OrderedDict (move to end on access); or DLL + hash map (key→node). *Brute force:* Store (key, value) in list; on get search list and move to end; on put add to end, if over capacity remove from front. O(n) get/put. *Optimization:* DLL for order (front=LRU, back=MRU); hash map key→node for O(1) lookup. get: lookup, move node to back, return value. put: if exists update and move to back; else add to back, evict front if over capacity. *Trade-off:* O(1) get/put, O(capacity) space. **O(1) get/put, O(capacity)**

---

## Trees

### 33. Invert Binary Tree
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Recursive (swap children then recurse); or BFS/DFS iterative with stack/queue. *Brute force:* Build a new tree that's the mirror (copy with left/right swapped). O(n) time, O(n) space. *Optimization:* In-place: swap left and right at root, then invert(left) and invert(right). Base case: null. *Trade-off:* O(n) time; recursion O(h) space, iterative with queue O(w) space. **O(n), O(h)**

---

### 34. Maximum Depth of Binary Tree
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Recursive (1 + max(left_depth, right_depth)); or BFS level count. *Brute force:* Not really — we have to visit each node. *Optimization:* DFS: if null return 0; else return 1 + max(depth(left), depth(right)). BFS: count layers until queue empty. *Trade-off:* O(n) time; recursion O(h) stack, BFS O(w) queue. **O(n), O(h)**

---

### 35. Diameter of Binary Tree
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* For each node compute "longest path through this node" (L_height + R_height + 1); or DP on tree. *Brute force:* For each pair of nodes find path length (LCA then distance). O(n²) or worse. *Optimization:* DFS that returns height. At each node, diameter through node = 1 + left_height + right_height. Update global max. Return height = 1 + max(L, R). *Trade-off:* One post-order pass O(n), O(h) recursion. **O(n), O(h)**

---

### 36. Validate BST
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Pass valid range (lo, hi) down; or inorder traversal and check ascending. *Brute force:* For each node, check all ancestors (or check left subtree max < root < right subtree min). O(n) per node ⇒ O(n²). *Optimization:* DFS(root, lo, hi): root must be in (lo, hi). Recurse left with (lo, root.val), right with (root.val, hi). Inorder: check each value > previous. *Trade-off:* O(n) time, O(h) space. Range approach is one pass; inorder needs to track previous. **O(n), O(h)**

---

### 37. Binary Tree Level Order Traversal
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* BFS with queue (process by level); or DFS with level index (result[level].append(val)). *Brute force:* BFS is the natural approach; "brute" would be track depth per node and group. *Optimization:* BFS: queue; for each "wave" (current queue length), pop that many nodes into one level list, add their children to queue. DFS: pass level; ensure result has list for that level, append node.val. *Trade-off:* BFS O(n) time, O(w) space; DFS O(n) time, O(h) stack. **O(n), O(w)**

---

### 38. Binary Tree Maximum Path Sum
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* At each node compute "best path through this node" (both branches) and "best path going up from this node" (one branch). *Brute force:* For each node as "top" of path, try all downward paths. O(n²) in worst case. *Optimization:* DFS returns max sum of a path going down from node (one branch). Path through node = node + max(0, left) + max(0, right). Update global max. Return to parent = node + max(left, right) (can't take both for the upward path). *Trade-off:* O(n) time, O(h) space. **O(n), O(h)**

---

## Heap

### 39. Kth Largest Element in Array
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Sort and take kth from end O(n log n); min-heap of size k O(n log k); QuickSelect O(n) average. *Brute force:* Sort, return nums[-k]. O(n log n). *Optimization:* Min-heap of size k: push each num, if size>k pop min. The kth largest is the smallest in this heap (heap top). QuickSelect: partition around pivot, recurse on side that contains kth largest. *Trade-off:* Heap O(n log k) time, O(k) space; QuickSelect O(n) avg but O(n²) worst case, O(1) extra. **O(n log k) heap, O(n) QuickSelect**

---

### 40. Merge K Sorted Lists
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Min-heap of k heads; or merge two at a time (1+2, then +3, ...) O(kN); or divide and conquer merge. *Brute force:* Collect all values, sort, build new list. O(N log N). Merge pairs: O(k) passes, each O(N). *Optimization:* Min-heap of size k: push (node.val, list_id, node) for each head. Pop smallest, append to result, push that list's next. *Trade-off:* O(N log k) time, O(k) heap space. D&C merge also O(N log k) with less heap overhead. **O(N log k)**

---

### 41. Find Median from Data Stream
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Two heaps (max-heap left, min-heap right); or sort on every median call (expensive); or balanced BST. *Brute force:* Store numbers in list; addNum append O(1); findMedian sort and pick middle O(n log n). *Optimization:* Left half in max-heap, right half in min-heap; keep sizes equal or left has one more. Median = top of max-heap (if sizes equal, or left larger) or average of both tops. On add, push then rebalance so size diff ≤ 1. *Trade-off:* addNum O(log n), findMedian O(1), O(n) space. **O(log n) add, O(1) median, O(n)**

---

## Backtracking

### 42. Subsets
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Backtrack (include/exclude each element); or iterative (for each existing subset, add next element). *Brute force:* Generate all 2^n bit masks, for each mask build subset. O(n·2^n). *Optimization:* Backtrack: at index i, we have a current path. Option 1: don't include nums[i], recurse(i+1). Option 2: include nums[i], path.append, recurse(i+1), path.pop(). Every call appends current path to result. *Trade-off:* O(n·2^n) time (2^n subsets, each up to n), O(n) recursion. **O(n·2^n), O(n)**

---

### 43. Combination Sum
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Backtrack with reuse (same index can be used again); or DP for count only. *Brute force:* Try all subsets of indices with replacement; filter to those summing to target. Very expensive. *Optimization:* Backtrack: at index i, we can include candidates[i] (stay at i for reuse) or skip (i+1). If total==target, record path. If total>target or i>=n, return. *Trade-off:* Time depends on target and candidates; space O(target) recursion. **O(2^target), O(target)**

---

### 44. Permutations
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Backtrack (pick one unused, recurse, unpick); or iterative (build from permutations of first k). *Brute force:* Generate all n! orderings (e.g. by recursion). *Optimization:* Backtrack: path so far; if len(path)==n, it's a permutation. For each num not yet in path, add it, recurse, remove. Use a "used" array for O(1) check. *Trade-off:* O(n·n!) time (n! permutations, each length n), O(n) space. **O(n·n!), O(n)**

---

### 45. Word Search
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* DFS from every cell; or build trie of words and explore board (for Word Search II). *Brute force:* For each (r,c) as start, DFS with "used" set; try all 4 directions. Backtrack: unmark when returning. *Optimization:* Same idea: DFS(r,c, index). Match board[r][c]==word[index]. Mark board[r][c] as used (or use a separate visited), recurse 4 dirs, unmark. Prune when char doesn't match. *Trade-off:* O(m·n·4^L) worst case; space O(L) recursion. **O(m·n·4^L), O(L)**

---

## Graphs

### 46. Number of Islands
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* DFS (flip '1' to '0' or use visited); or BFS. Union-Find also works. *Brute force:* For each '1', do a DFS/BFS to mark entire island (e.g. flip to '0'). Count how many times we start a new island. *Optimization:* Same: for each unvisited '1', run DFS/BFS marking all connected '1's, increment count. Avoid extra visited by mutating grid to '0'. *Trade-off:* O(m·n) time; DFS stack O(m·n) worst, BFS queue O(m·n). **O(m·n), O(m·n)**

---

### 47. Clone Graph
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* DFS with hash map (node→clone); or BFS. Map avoids infinite loop and re-cloning. *Brute force:* Create a copy per node without map → we'd clone same node many times and cycles never end. *Optimization:* Map original→clone. DFS: if node in map return map[node]. Else create clone, map[node]=clone, clone.neighbors = [dfs(neighbor) for neighbor in node.neighbors]. *Trade-off:* O(V+E) time, O(V) space for map and clones. **O(V+E), O(V)**

---

### 48. Course Schedule (Cycle Detection)
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* DFS with three states (detect back edge); or Kahn's (BFS with in-degree, if we can't process all nodes there's a cycle). *Brute force:* Try to topo-sort; if we can't (no node with in-degree 0 and nodes left), cycle. *Optimization:* DFS: state 0=unvisited, 1=visiting, 2=done. If we hit a node with state 1, we have a back edge → cycle. Recurse neighbors, then set state 2. *Trade-off:* O(V+E) time, O(V) space. **O(V+E), O(V)**

---

### 49. Word Ladder
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* BFS (shortest path in graph where edges = one-letter change); or bidirectional BFS. *Brute force:* BFS: state = (word, steps). From each word, try all one-letter changes; if in wordList, add to queue. First time we hit endWord, return steps. *Optimization:* Same. Represent wordList as set for O(1) lookup and remove used words (so we don't revisit). Each word has M chars, 26 options per char ⇒ O(M·26) neighbors. *Trade-off:* O(M²·N) for N words in list, M length. Bidirectional can cut constant. **O(M²·N)**

---

## Dynamic Programming (1-D)

### 50. Climbing Stairs
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* DP (recurrence); or just Fibonacci — same as "ways to reach step i". *Brute force:* Recursion: climb(i) = climb(i-1) + climb(i-2) with base cases. O(2^n) without memo. *Optimization:* dp[i] = dp[i-1] + dp[i-2]; dp[0]=1, dp[1]=1. Or three variables: prev2, prev1, curr; roll forward. *Trade-off:* O(n) time, O(1) space with variables. **O(n), O(1)**

---

### 51. House Robber
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* DP (at each house, rob or skip); or state machine (robbed_prev, not_robbed_prev). *Brute force:* Try all 2^n subsets of houses (no two adjacent); take max sum. O(2^n). *Optimization:* dp[i] = max money from first i houses. Option 1: skip i → dp[i-1]. Option 2: rob i → nums[i-1] + dp[i-2]. So dp[i] = max(dp[i-1], nums[i-1]+dp[i-2]). O(1) space: two vars. *Trade-off:* O(n) time, O(1) space. **O(n), O(1)**

---

### 52. Coin Change
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* DP (min coins for amount a); or BFS (state = amount, edges = subtract a coin). *Brute force:* Recursion: try each coin, recurse on amount - coin; take min. O(amount^coins) without memo. *Optimization:* dp[a] = min coins to make amount a. dp[0]=0. For a from 1 to amount, dp[a] = 1 + min(dp[a-c] for c in coins if a>=c). If no valid coin, dp[a]=inf. *Trade-off:* O(amount · len(coins)) time, O(amount) space. **O(amount·len(coins)), O(amount)**

---

### 53. Longest Increasing Subsequence
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* DP O(n²) (for each i, look at all j<i); or patience sorting / binary search O(n log n) (maintain smallest tail of length-L LIS). *Brute force:* Try all 2^n subsequences, check increasing, take max length. O(2^n). *Optimization:* dp[i] = length of LIS ending at i. dp[i] = 1 + max(dp[j] for j<i if nums[j]<nums[i]). Return max(dp). Better: maintain array "tails"; for each num, binary search where it fits, update. *Trade-off:* DP O(n²), O(n); patience O(n log n), O(n). **O(n²) or O(n log n)**

---

### 54. Word Break
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* DP (can we segment s[0:i]?); or backtrack (try break at each position). *Brute force:* Recursion: try breaking at each position, if prefix in dict recurse on rest. O(2^n) without memo. *Optimization:* dp[i] = True if s[0:i] can be segmented. dp[0]=True. For i, dp[i] = any(dp[j] and s[j:i] in wordDict for j in 0..i-1). Use set for wordDict. *Trade-off:* O(n² · m) if we do string slice; can optimize with trie. O(n) space. **O(n²·m), O(n)**

---

## Greedy

### 55. Maximum Subarray (Kadane)
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Kadane (max sum ending at i); or divide and conquer (max in left, right, or crossing). *Brute force:* For each (i,j), sum subarray, take max. O(n²) or O(n³). *Optimization:* Kadane: cur = max sum of subarray ending at current element = max(num, cur+num). max_sum = max(max_sum, cur). If cur+num < num, we restart at num. *Trade-off:* O(n) time, O(1) space. **O(n), O(1)**

---

### 56. Jump Game
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* DP (can we reach index i?); or greedy (furthest we can reach). *Brute force:* Try all jump choices at each step (branching). Exponential. *Optimization:* Greedy: track max_reach (furthest index we can get to so far). For each i, if i > max_reach we're stuck → false. Else max_reach = max(max_reach, i + nums[i]). If max_reach >= n-1 → true. *Trade-off:* O(n) time, O(1) space. We only need to know "can we get past here?" not path. **O(n), O(1)**

---

### 57. Merge Intervals
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Sort by start then merge in one pass; or sort by end (different merge logic). *Brute force:* For each interval, check all others for overlap, merge, repeat until no changes. O(n²) or more. *Optimization:* Sort by start. res = [first]. For each next: if next.start <= res[-1].end they overlap → merge (res[-1].end = max(res[-1].end, next.end)); else append. *Trade-off:* O(n log n) for sort, O(n) space. **O(n log n), O(n)**

---

## Bit Manipulation

### 58. Single Number
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* XOR all (pairs cancel); or hash map count, return key with count 1; or sum and math. *Brute force:* Count each number (hash map), return the one with count 1. O(n) time, O(n) space. *Optimization:* XOR is commutative and associative. a^a=0, a^0=a. So xor of all numbers = xor of (pairs of same) + (single) = 0 + single = single. *Trade-off:* O(n) time, O(1) space. **O(n), O(1)**

---

### 59. Number of 1 Bits
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Shift and count (n&1, n>>=1); or clear lowest set bit with n &= n-1 until 0. *Brute force:* Loop 32 times, count += (n>>i)&1. O(1) for fixed width. *Optimization:* n & (n-1) flips the rightmost 1 to 0. So while n: n &= n-1; count++. Number of iterations = number of 1s. *Trade-off:* O(1) for 32-bit (at most 32 iterations). **O(1)** (32 bits)

---

### 60. Counting Bits
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* For each i count 1s (e.g. n&=n-1) O(n·bits); or DP: ans[i] from ans[i>>1] or ans[i&(i-1)]. *Brute force:* For i 0..n, count set bits in i (loop or popcount). O(n log n) or O(n·32). *Optimization:* ans[i] = ans[i>>1] + (i&1). Number of 1s in i = number in i/2 (i>>1) plus LSB. Or ans[i] = ans[i & (i-1)] + 1 (drop rightmost 1). *Trade-off:* O(n) time, O(1) extra space (output array doesn't count). **O(n), O(1)**

---

## Trees (continued)

### 61. Balanced Binary Tree
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* For each node compute height of left and right and check |L-R|≤1 (can get O(n) if we return (height, balanced) to avoid recomputing). *Brute force:* At each node recursively compute height of left and right subtrees; if |diff|>1 return false. Without caching, height is recomputed many times → O(n²). *Optimization:* DFS returns (height, is_balanced). If either subtree not balanced, or |left_h - right_h| > 1, return false. *Trade-off:* Single post-order pass O(n), O(h) stack. **O(n), O(h)**

---

### 62. Same Tree
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Recursive (compare roots then left and right); or iterative BFS/DFS comparing level by level. *Brute force:* Serialize both trees and compare strings. O(n) but extra space. *Optimization:* If both null → true; if one null → false; if val differ → false; return sameTree(left,left) and sameTree(right,right). *Trade-off:* O(n) time, O(h) recursion. **O(n), O(h)**

---

### 63. Subtree of Another Tree
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* For each node of root, check if subtree there equals subRoot (sameTree); or serialize and check if subRoot's serialization is substring (KMP). *Brute force:* For every node in root, run sameTree(node, subRoot). O(n·m) where m = size of subRoot. *Optimization:* DFS: at each node, if sameTree(node, subRoot) return true; else recurse left and right. *Trade-off:* O(n·m) worst case; serialization + KMP can be O(n+m). **O(n·m), O(h)**

---

### 64. Lowest Common Ancestor of BST
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Use BST property: LCA is where one target is in left subtree and other in right, or one is the node. *Brute force:* Find path to p and path to q, then find last common node. O(n) time and space. *Optimization:* If both p,q < root, LCA in left; if both > root, LCA in right; else root is LCA. Recurse accordingly. O(h) time. *Trade-off:* O(h) time, O(h) recursion; iterative uses O(1) space. **O(h), O(h)**

---

### 65. Binary Tree Right Side View
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* BFS and take last node of each level; or DFS (visit right first) and record first time we see each depth. *Brute force:* Level order, for each level take the last element. *Optimization:* BFS: for each level size, pop that many; the last pop is rightmost. DFS: recurse root.right then root.left; when depth == len(result), append (first time at this depth = rightmost). *Trade-off:* O(n) time, O(w) BFS or O(h) DFS. **O(n), O(w)**

---

### 66. Count Good Nodes in Binary Tree
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* DFS passing max value from root to current; if node.val >= max so far, it's good. *Brute force:* For each node, check path to root that all values ≤ node.val. O(n·h). *Optimization:* DFS(root, max_so_far): if not root return 0; good = 1 if root.val >= max_so_far else 0; new_max = max(max_so_far, root.val); return good + dfs(left, new_max) + dfs(right, new_max). *Trade-off:* O(n) time, O(h) space. **O(n), O(h)**

---

### 67. Kth Smallest Element in BST
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Inorder traversal (gives sorted order), take kth; or augment tree with rank. *Brute force:* Inorder into array, return arr[k-1]. O(n) time, O(n) space. *Optimization:* Inorder but stop when we've seen k nodes (recursive or iterative with stack). *Trade-off:* O(h + k) with early stop; O(n) worst. **O(h+k), O(h)**

---

### 68. Construct Binary Tree from Preorder and Inorder
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Root = preorder[0]; find root in inorder; left subtree = left part of inorder, right = right part; recurse with matching preorder chunks. *Brute force:* Same idea but search for root in inorder linearly each time. O(n²). *Optimization:* Build map inorder value → index. Root = pre[0]; split inorder at root; preorder for left = pre[1:1+len_left], right = pre[1+len_left:]. Recurse. *Trade-off:* O(n) time with map, O(n) space. **O(n), O(n)**

---

### 69. Serialize and Deserialize Binary Tree
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Preorder with "null" for missing children; or level order. *Brute force:* Preorder: serialize as "val,left,right" with "N" for null. Deserialize: split by comma, consume one token at a time, build tree recursively. *Optimization:* Same. Use a queue or index to consume tokens during deserialize. *Trade-off:* O(n) both ways, O(n) space for encoded string and recursion. **O(n), O(n)**

---

## Tries

### 70. Implement Trie (Prefix Tree)
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Array of 26 children per node; or hash map for children. *Brute force:* Store all strings in a list; search/startWith by scanning. O(n·k) per op. *Optimization:* Each node has children[26] and is_end. Insert: walk character by character, create nodes as needed, mark end. Search: walk and check is_end at end. startsWith: walk and return true if we can complete. *Trade-off:* O(m) per op (m = word length), O(n·k) space for n words of length up to k. **O(m), O(n·k)**

---

### 71. Design Add and Search Words Data Structure
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Trie; on search with '.', try all 26 children at that level. *Brute force:* Store words in list; search by matching with wildcard. O(n·k). *Optimization:* Same trie; addWord is standard. search(word): if char is '.', recurse on all non-null children; else recurse on child[char]. *Trade-off:* Add O(m), search O(26^d) worst if many '.'; space O(n·k). **O(m) add, O(26^m) search worst**

---

### 72. Word Search II
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* For each word run Word Search I (DFS) — slow; or build trie of words, then DFS board and walk trie. *Brute force:* For each word, search board (Word Search I). O(words · m·n·4^L). *Optimization:* Put all words in trie. For each cell, DFS and follow trie; when we hit a word end, add to result and optionally remove from trie to avoid duplicates. Prune when trie node has no children. *Trade-off:* O(m·n·4^L) but with trie pruning; space O(total chars in words). **O(m·n·4^L), O(W)**

---

## Heap (continued)

### 73. Kth Largest Element in a Stream
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Min-heap of size k (keep k largest); or sort on each add (expensive). *Brute force:* Store all numbers, on add sort and return kth from end. O(n log n) per add. *Optimization:* Min-heap of size k. add: push val; if len>k pop min. The kth largest is the minimum in the heap (top). *Trade-off:* add O(log k), O(k) space. **O(log k) add, O(k)**

---

### 74. Last Stone Weight
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Max-heap (use negative for Python heapq); each step pop two, push difference. *Brute force:* Repeatedly sort, take two largest, replace with difference. O(n² log n). *Optimization:* Heapify array (max-heap). While len>1: pop two (a,b), if a!=b push a-b. Return last remaining or 0. *Trade-off:* O(n log n) time, O(n) space. **O(n log n), O(n)**

---

### 75. K Closest Points to Origin
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Sort by distance O(n log n); or max-heap of size k (keep k closest). *Brute force:* Sort points by distance, return first k. O(n log n). *Optimization:* Max-heap of size k (key = distance). For each point: if heap size < k push; else if dist < heap max, pop and push. Return heap. *Trade-off:* O(n log k) time, O(k) space. **O(n log k), O(k)**

---

### 76. Task Scheduler
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Greedy: schedule most frequent first with n gaps; or count idle slots. *Brute force:* Try all orderings — infeasible. *Optimization:* Count freq; max_freq = max. We need at least (max_freq-1)*(n+1) + (number of tasks with max_freq) slots, or len(tasks) if that's larger. *Trade-off:* O(n) time, O(26) space for counts. **O(n), O(1)**

---

### 77. Design Twitter
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* For feed: get followees' recent tweets and merge (heap); store userId → list of (time, tweetId). *Brute force:* getNewsFeed: collect all tweets from followees, sort by time, return top 10. O(total tweets log total). *Optimization:* Each user has a list of (timestamp, tweetId). getNewsFeed: heap of (time, tweetId, list_idx, next_idx) from each followee's list; pop 10. *Trade-off:* post O(1), getNewsFeed O(10 log followees), follow/unfollow O(1). **O(1) post, O(k log n) feed**

---

## Backtracking (continued)

### 78. Subsets II
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Same as Subsets but skip duplicate branches: sort, then when we skip an element, skip all same elements. *Brute force:* Generate all 2^n subsets, dedupe. O(n·2^n). *Optimization:* Sort nums. Backtrack(i): append path. For j in range(i, n): if j>i and nums[j]==nums[j-1], continue; path.append(nums[j]), backtrack(j+1), path.pop(). *Trade-off:* O(n·2^n) time, O(n) space. **O(n·2^n), O(n)**

---

### 79. Combination Sum II
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Like Combination Sum but no reuse; sort and skip duplicate values at same depth. *Brute force:* Try all subsets, filter sum==target, dedupe. *Optimization:* Sort. Backtrack(i, path, total): if total==target append; if total>target or i>=n return. For j from i: if j>i and candidates[j]==candidates[j-1] skip; include candidates[j], backtrack(j+1), pop. *Trade-off:* O(2^n) in practice, O(n) space. **O(2^n), O(n)**

---

### 80. Palindrome Partitioning
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Backtrack: at each position try cutting (if prefix is palindrome, recurse on rest). *Brute force:* Try all 2^(n-1) partitions, keep those with all palindromic parts. *Optimization:* backtrack(start): if start==len(s) append path. For end from start to n: if s[start:end+1] is palindrome, path.append, backtrack(end+1), path.pop(). Precompute isPal[i][j] optional. *Trade-off:* O(n·2^n) worst, O(n) recursion. **O(n·2^n), O(n)**

---

### 81. Letter Combinations of a Phone Number
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Backtrack: for each digit, try each letter, recurse on next digit. *Brute force:* Same — we must try 4^d (for digit 9) combinations. *Optimization:* Map digit → "abc". path = []; backtrack(i): if i==len(digits) append ''.join(path). For c in map[digits[i]]: path.append(c), backtrack(i+1), path.pop(). *Trade-off:* O(4^n) time (n = len digits), O(n) space. **O(4^n), O(n)**

---

### 82. N-Queens
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Backtrack: place queen row by row; at each row try each col that doesn't conflict with previous. *Brute force:* Try all n^n placements, check valid. *Optimization:* Track cols, diag1 (r+c), diag2 (r-c) used. backtrack(r): if r==n append board. For c: if col c and both diags free, place queen, set used, backtrack(r+1), unset. *Trade-off:* O(n!) in practice (pruning), O(n²) space for board. **O(n!), O(n²)**

---

### 83. N-Queens II
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Same as N-Queens but count solutions instead of storing boards. *Brute force:* Same backtracking. *Optimization:* Same backtrack; increment count on base case instead of appending. Can use bitsets for cols/diags to save space. *Trade-off:* O(n!) time, O(n) space (no board storage). **O(n!), O(n)**

---

## Graphs (continued)

### 84. Pacific Atlantic Water Flow
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* From each cell DFS to both oceans — slow; or DFS from Pacific border (all reachable) and from Atlantic border, intersect. *Brute force:* For each cell, check if can reach Pacific and Atlantic. O((m·n)²). *Optimization:* Two visited sets. DFS from all Pacific-edge cells (left + top); DFS from all Atlantic-edge cells (right + bottom). Result = cells in both sets. *Trade-off:* O(m·n) time, O(m·n) space. **O(m·n), O(m·n)**

---

### 85. Surrounded Regions
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Flip all O to X then restore O's that are connected to border (DFS from border). *Brute force:* For each O, check if surrounded by X (flood fill to border). O(n²) per O. *Optimization:* DFS from every border O, mark reachable O's (e.g. to 'T'). Then flip all remaining O to X and T back to O. *Trade-off:* O(m·n) time, O(m·n) recursion. **O(m·n), O(m·n)**

---

### 86. Rotting Oranges
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* BFS from all rotten oranges at once (multi-source); each minute = one level. *Brute force:* Repeatedly scan grid, rot neighbors of rotten; repeat until no change. O(minutes · m·n). *Optimization:* Queue = initial rotten cells. BFS: for each level, pop all, add fresh neighbors to queue and mark rotten. If fresh left at end return -1. *Trade-off:* O(m·n) time and space. **O(m·n), O(m·n)**

---

### 87. Course Schedule II
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Topological sort (Kahn's or DFS); if cycle, return []. *Brute force:* Same — we need a valid order. *Optimization:* Build graph. Kahn: in-degrees, queue of 0-in-degree; process and reduce neighbors' in-degree. Order = processing order. Or DFS: post-order push to stack, reverse = topo order; detect cycle with three states. *Trade-off:* O(V+E) time, O(V) space. **O(V+E), O(V)**

---

### 88. Redundant Connection
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Union-Find: add edges one by one; first edge that connects already-connected nodes is the answer. *Brute force:* For each edge, check if graph without it is still connected. O(E²). *Optimization:* Union-Find. For edge (u,v): if find(u)==find(v) return this edge (redundant); else union(u,v). *Trade-off:* O(E·α(n)) time, O(n) space. **O(E·α(n)), O(n)**

---

### 89. Number of Connected Components
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Union-Find then count distinct roots; or DFS/BFS from each unvisited node, count components. *Brute force:* DFS from each node that hasn't been visited; increment count each time we start. *Optimization:* Union-Find: union all edges, then count distinct find(i). Or DFS: for i in range(n), if not visited, dfs(i), count++. *Trade-off:* O(n+E) time, O(n) space. **O(n+E), O(n)**

---

### 90. Graph Valid Tree
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Tree ⟺ connected and exactly n-1 edges. Check edge count then BFS/DFS to see if one component. *Brute force:* If edges != n-1 return false; else check connectivity. *Optimization:* if len(edges) != n-1: return False. Build graph, BFS/DFS from 0; if we visit exactly n nodes, connected. *Trade-off:* O(n+E) time, O(n) space. **O(n+E), O(n)**

---

### 91. Word Ladder II
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* BFS to find shortest length; then DFS to enumerate paths of that length (backtrack from endWord). *Brute force:* BFS and store all paths — memory blow-up. *Optimization:* BFS from beginWord, build graph (word → list of next words) and stop at depth when we see endWord. Then DFS from beginWord to endWord with that graph to build paths. Or reverse BFS from endWord to get parents, then DFS to build paths. *Trade-off:* O(n·k²) for building graph, O(2^n) paths worst. **O(n·k²), O(paths)**

---

## Advanced Graphs

### 92. Reconstruct Itinerary
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Euler path: DFS, use each edge once; lexicographically smallest = sort neighbors and try smallest first. *Brute force:* Try all permutations of tickets. *Optimization:* Build graph: from → sorted list of to's. DFS(from): while from has unused edge, pop smallest to, DFS(to). Append 'from' after recursion (post-order). Reverse result. *Trade-off:* O(E log E) for sorting, O(E) DFS. **O(E log E), O(E)**

---

### 93. Min Cost to Connect All Points
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* MST: Prim or Kruskal. *Brute force:* Try all spanning trees — exponential. *Optimization:* Prim: start with one node, add closest unvisited point (min Manhattan distance) until all in. O(n²) for n points. Kruskal: all n(n-1)/2 edges, sort by weight, union-find add until n-1 edges. *Trade-off:* Prim O(n²), Kruskal O(n² log n) with sort. **O(n²), O(n)**

---

### 94. Network Delay Time
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Dijkstra from source k; return max of distances (or -1 if any unreachable). *Brute force:* BFS only works for unweighted. *Optimization:* Dijkstra: min-heap (time, node). Pop smallest time, if we haven't seen node, update dist[node], push (time+w, neighbor) for each edge. Return max(dist) or -1. *Trade-off:* O(E log V) with heap, O(V) space. **O(E log V), O(V)**

---

### 95. Swim in Rising Water
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Binary search on time t; check if path (0,0)→(n-1,n-1) exists with all cells ≤ t. Or Dijkstra on grid (weight = max(so far, grid[r][c])). *Brute force:* Try t = 0,1,... until path exists. *Optimization:* Binary search t; BFS/DFS only through cells with value ≤ t. Or use min-heap: state (max_height_so_far, r, c); expand to neighbors with new_max = max(current, grid[nr][nc]); first time we reach (n-1,n-1) return that max. *Trade-off:* Binary search O(n² log(max)), Dijkstra O(n² log(n²)). **O(n² log n), O(n²)**

---

### 96. Alien Dictionary
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Build graph: from adjacent words, first differing char gives edge (prev → next). Topological sort. *Brute force:* Not clear. *Optimization:* For consecutive words w1, w2: find first i where w1[i]!=w2[i], add edge w1[i]->w2[i]; if w2 is prefix of w1, invalid. Then topo sort. Handle isolated chars (add all chars seen). *Trade-off:* O(C + E) where C = total chars, E = edges from pairs. **O(C+E), O(1)** chars

---

### 97. Cheapest Flights Within K Stops
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* BFS with state (city, cost, stops); or Bellman-Ford (relax edges k+1 times). *Brute force:* Try all paths with ≤k+1 edges — exponential. *Optimization:* dp[stops][city] = min cost to reach city in exactly stops. Initialize dp[0][src]=0. For stops 1..k+1, for each (from,to,price), dp[stops][to] = min(dp[stops][to], dp[stops-1][from]+price). Or BFS with (city, cost), only extend if stops ≤ k. *Trade-off:* O(K·E) or O(E·K), O(V·K) or O(V). **O(K·E), O(V)**

---

## 1-D DP (continued)

### 98. Min Cost Climbing Stairs
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* dp[i] = min cost to reach step i (from 0 or 1); then min(dp[n-1], dp[n-2]) to reach top. *Brute force:* Recursion try step from 0 and from 1. O(2^n). *Optimization:* dp[i] = cost[i] + min(dp[i-1], dp[i-2]); dp[0]=cost[0], dp[1]=cost[1]. Answer = min(dp[-1], dp[-2]). O(1) space: two vars. *Trade-off:* O(n) time, O(1) space. **O(n), O(1)**

---

### 99. House Robber II
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Run House Robber twice: exclude first house, then exclude last house; take max. *Brute force:* Try all valid subsets with circular constraint. *Optimization:* rob(nums) = House Robber I. Return max(rob(nums[1:]), rob(nums[:-1])) — one of these covers the optimal (either we skip first or last). *Trade-off:* O(n) time, O(1) space. **O(n), O(1)**

---

### 100. Longest Palindromic Substring
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Expand around each center (2n-1 centers: char or between chars); or DP isPal[i][j]. *Brute force:* For each (i,j) check if s[i:j+1] is palindrome. O(n³). *Optimization:* For each center, expand while s[L]==s[R]. Track longest. Odd: center i; even: centers i and i+1. *Trade-off:* O(n²) time, O(1) space. **O(n²), O(1)**

---

### 101. Palindromic Substrings
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Same expand-around-center; count every palindrome. Or DP count. *Brute force:* For each (i,j) check palindrome, count. O(n³). *Optimization:* Expand around each center; for each valid (L,R) count++. *Trade-off:* O(n²) time, O(1) space. **O(n²), O(1)**

---

### 102. Decode Ways
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* dp[i] = ways to decode s[0:i]. Take one char (if 1-9) or two (if 10-26). *Brute force:* Recursion: decode(s) = (decode(s[1:]) if valid) + (decode(s[2:]) if valid). O(2^n). *Optimization:* dp[0]=1, dp[1]=1 if s[0]!='0'. dp[i] += dp[i-1] if s[i-1] in '1'-'9'; += dp[i-2] if 10<=int(s[i-2:i])<=26. *Trade-off:* O(n) time, O(1) space with vars. **O(n), O(1)**

---

### 103. Maximum Product Subarray
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Track max and min product ending here (negative can become max when multiplied). *Brute force:* For each (i,j) compute product. O(n²). *Optimization:* cur_max = cur_min = 1. For n: new_max = max(n, cur_max*n, cur_min*n); new_min = min(n, cur_max*n, cur_min*n); update cur_max, cur_min; ans = max(ans, cur_max). *Trade-off:* O(n) time, O(1) space. **O(n), O(1)**

---

### 104. Partition Equal Subset Sum
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Subset sum: can we make total/2? DP[sum] = achievable. *Brute force:* Try all 2^n subsets. *Optimization:* If total odd return false. dp = set([0]). For each num, dp = dp | {s+num for s in dp}. Return total//2 in dp. Or 1D boolean array dp[0..total/2]. *Trade-off:* O(n·sum) time, O(sum) space. **O(n·sum), O(sum)**

---

## Intervals

### 105. Insert Interval
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Add newInterval, then merge (sort and merge); or one pass: add all intervals that end before newInterval.start, merge overlapping with newInterval, add rest. *Brute force:* Append, sort by start, merge. O(n log n). *Optimization:* One pass: push intervals that end before newInterval.start; then while intervals start <= newInterval.end, merge into newInterval; push newInterval and remaining. *Trade-off:* O(n) time, O(1) extra if we build result. **O(n), O(n)**

---

### 106. Non-overlapping Intervals
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Greedy: sort by end; keep interval if start >= last_end, else remove (count removal). *Brute force:* DP or try all subsets of intervals. *Optimization:* Sort by end. last_end = -inf. For each (s,e): if s >= last_end keep (last_end = e); else remove (count++). We remove the one that ends later to leave more room. *Trade-off:* O(n log n) time, O(1) extra. **O(n log n), O(1)**

---

### 107. Meeting Rooms
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Sort by start; check intervals[i].end <= intervals[i+1].start. *Brute force:* Check all pairs for overlap. O(n²). *Optimization:* Sort by start. For i in range(len-1): if intervals[i][1] > intervals[i+1][0] return False. *Trade-off:* O(n log n) time, O(1) space. **O(n log n), O(1)**

---

### 108. Meeting Rooms II
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Min-heap of end times; or sort starts and ends, sweep and count. *Brute force:* For each time point count how many intervals contain it. O(n·range). *Optimization:* Sort start times and end times. Two pointers: when start[i] < end[j] we need a new room (count++); else free a room (j++). Return max count. Or heap: sort by start; for each meeting, if min(end) <= start pop; push end. *Trade-off:* O(n log n) time, O(n) space. **O(n log n), O(n)**

---

### 109. Minimum Interval to Include Each Query
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* For each query, find all intervals containing it, take min size — slow. Sort intervals by left; for each query, consider intervals that started before query and haven't ended; min-heap by size. *Brute force:* For each query scan all intervals. O(Q·n). *Optimization:* Sort intervals by left. Sort queries with original index. Sweep: add intervals that start <= query to heap (key = (size, right)); pop until top's right >= query; answer[query_idx] = top size or -1. *Trade-off:* O(n log n + Q log Q + (n+Q) log n), O(n). **O((n+Q) log n), O(n)**

---

## Greedy (continued)

### 110. Jump Game II
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* BFS (each level = one jump); or greedy: track current reach and next reach; when we pass current reach, jump++ and current = next. *Brute force:* Try all jump sequences. Exponential. *Optimization:* jumps = 0, cur_end = 0, farthest = 0. For i in range(n-1): farthest = max(farthest, i+nums[i]). If i == cur_end: jumps++, cur_end = farthest. *Trade-off:* O(n) time, O(1) space. **O(n), O(1)**

---

### 111. Gas Station
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* If total gas >= total cost, solution exists. Try start from 0; if we run out at i, try start from i+1 (can't start from 1..i). *Brute force:* Try each index as start, simulate. O(n²). *Optimization:* total_gas - total_cost: if < 0 return -1. tank = 0, start = 0. For i: tank += gas[i]-cost[i]; if tank < 0, start = i+1, tank = 0. Return start. *Trade-off:* O(n) time, O(1) space. **O(n), O(1)**

---

### 112. Hand of Straights
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Sort and greedy: for each card, if it starts a new group need groupSize consecutive; use Counter and try to form groups. *Brute force:* Try all partitions. *Optimization:* Count cards. For each card in sorted order: if count[card]==0 continue. For k in range(groupSize): if count[card+k] < count[card] can't form group; else count[card+k] -= count[card] (or subtract 1 each time we form one group). Actually: for each card, while we have this card, form group starting at card (card, card+1, ..., card+groupSize-1), decrement counts. *Trade-off:* O(n log n) sort, O(n) space. **O(n log n), O(n)**

---

### 113. Partition Labels
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* For each char, we need its last index in the partition. Greedy: extend partition until we've included all last indices of chars in it. *Brute force:* Try all split points. *Optimization:* last = {c: i for i,c in enumerate(s)}. start = end = 0; for i,c in enumerate(s): end = max(end, last[c]); if i == end: append end-start+1, start = i+1. *Trade-off:* O(n) time, O(1) space (26 chars). **O(n), O(1)**

---

### 114. Valid Parenthesis String
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Track range of possible open count: low = treat '*' as ')', high = treat '*' as '('. *Brute force:* Try treat each '*' as '(', ')', or empty. O(3^n). *Optimization:* low, high = 0, 0. For c: if '(', low++, high++; if ')', low--, high--; if '*', low--, high++. If high < 0 break. low = max(low, 0). Return low == 0. *Trade-off:* O(n) time, O(1) space. **O(n), O(1)**

---

## 2-D Dynamic Programming

### 115. Unique Paths
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* dp[i][j] = paths to (i,j) = dp[i-1][j] + dp[i][j-1]; or math: C(m+n-2, n-1). *Brute force:* DFS from (0,0) count paths. O(2^(m+n)). *Optimization:* dp[0][*]=1, dp[*][0]=1. dp[i][j]=dp[i-1][j]+dp[i][j-1]. O(m·n) time; O(n) space with one row. *Trade-off:* O(m·n) time, O(min(m,n)) space. **O(m·n), O(n)**

---

### 116. Longest Common Subsequence
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* dp[i][j] = LCS of text1[0:i], text2[0:j]. If text1[i-1]==text2[j-1] then 1+dp[i-1][j-1]; else max(dp[i-1][j], dp[i][j-1]). *Brute force:* Try all subsequences of both. O(2^(n+m)). *Optimization:* 2D table; fill row by row. Can reduce to 2 rows. *Trade-off:* O(n·m) time, O(min(n,m)) space. **O(n·m), O(min(n,m))**

---

### 117. Best Time to Buy and Sell Stock with Cooldown
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* State machine: held, sold, rest. rest = max(rest, sold); held = max(held, rest - price); sold = held + price. *Brute force:* Try all buy/sell sequences with cooldown. *Optimization:* rest[i] = max(rest[i-1], sold[i-1]); held[i] = max(held[i-1], rest[i-1]-p); sold[i] = held[i-1]+p. O(1) space with three vars. *Trade-off:* O(n) time, O(1) space. **O(n), O(1)**

---

### 118. Coin Change II
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* dp[amount] = number of combinations. For each coin, dp[a] += dp[a-coin]. *Brute force:* Backtrack count. *Optimization:* dp[0]=1. For each coin c, for a from c to amount: dp[a] += dp[a-c]. Order of loops matters (coins outer = one way to get each combination). *Trade-off:* O(amount·len(coins)) time, O(amount) space. **O(amount·coins), O(amount)**

---

### 119. Target Sum
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Subset sum: we need (sum_pos - sum_neg) = target; with sum_pos + sum_neg = total, so sum_pos = (total+target)/2. Count subsets with that sum. *Brute force:* Try all +/-. O(2^n). *Optimization:* If (total+target) odd or target > total return 0. dp[sum] = ways to make sum. dp[0]=1; for each num, for s from target down to num: dp[s] += dp[s-num]. *Trade-off:* O(n·sum) time, O(sum) space. **O(n·sum), O(sum)**

---

### 120. Interleaving String
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* dp[i][j] = can we form s3[0:i+j] from s1[0:i] and s2[0:j]. dp[i][j] = (s1[i-1]==s3[i+j-1] and dp[i-1][j]) or (s2[j-1]==s3[i+j-1] and dp[i][j-1]). *Brute force:* Recursion try take from s1 or s2. O(2^(n+m)). *Optimization:* 2D DP; can use 1D row. *Trade-off:* O(n·m) time, O(min(n,m)) space. **O(n·m), O(m)**

---

### 121. Longest Increasing Path in a Matrix
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* DFS from each cell with memo: lip(r,c) = 1 + max(lip(nr,nc) for neighbors with value > matrix[r][c]). *Brute force:* DFS from each cell without memo. O((m·n)²). *Optimization:* dp[r][c] = result of lip(r,c). If computed return; else dp[r][c] = 1 + max(dfs(nr,nc) for valid neighbors with larger value). *Trade-off:* O(m·n) time, O(m·n) space. **O(m·n), O(m·n)**

---

### 122. Distinct Subsequences
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* dp[i][j] = number of ways s[0:i] has subsequence t[0:j]. If s[i-1]==t[j-1]: dp[i][j] = dp[i-1][j-1] + dp[i-1][j]; else dp[i][j] = dp[i-1][j]. *Brute force:* Try all subsequences of s matching t. *Optimization:* 2D table; can use 1D (scan j backwards). *Trade-off:* O(n·m) time, O(m) space. **O(n·m), O(m)**

---

### 123. Edit Distance
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* dp[i][j] = min ops to convert word1[0:i] to word2[0:j]. Insert/delete/replace. *Brute force:* Recursion try all ops. O(3^(n+m)). *Optimization:* dp[i][j] = min(1+dp[i-1][j], 1+dp[i][j-1], (0 if word1[i-1]==word2[j-1] else 1)+dp[i-1][j-1]). Base: dp[i][0]=i, dp[0][j]=j. *Trade-off:* O(n·m) time, O(min(n,m)) space. **O(n·m), O(min(n,m))**

---

### 124. Burst Balloons
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* DP: dp[i][j] = max coins from bursting balloons i..j (with virtual 1 on sides). Last burst = k in [i,j], coins = nums[i-1]*nums[k]*nums[j+1] + dp[i][k-1] + dp[k+1][j]. *Brute force:* Try all orders. O(n!). *Optimization:* Fill dp for length 1..n. Pad nums with 1 at ends. *Trade-off:* O(n³) time, O(n²) space. **O(n³), O(n²)**

---

### 125. Regular Expression Matching
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* dp[i][j] = does s[0:i] match p[0:j]. Handle '.' and '*' (zero or more of preceding). *Brute force:* Recursion with memo. *Optimization:* If p[j-1]=='*': match zero (dp[i][j-2]) or match one (if s[i-1]==p[j-2] or p[j-2]=='.') and dp[i-1][j]. Else: (s[i-1]==p[j-1] or p[j-1]=='.') and dp[i-1][j-1]. *Trade-off:* O(n·m) time, O(n·m) space. **O(n·m), O(n·m)**

---

## Bit Manipulation (continued)

### 126. Reverse Bits
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Extract bit by bit and build result; or swap in pairs (reverse 16-bit halves, then 8-bit, etc.). *Brute force:* For i in 0..31, result |= (n>>i & 1) << (31-i). O(32). *Optimization:* Same. Or divide and conquer: swap halves, then each half's halves, etc. *Trade-off:* O(1) for fixed 32 bits. **O(1), O(1)**

---

### 127. Missing Number
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Sum 0..n - sum(nums); or XOR 0..n with XOR of nums (same as Single Number). *Brute force:* Sort and find gap; or set of 0..n minus set(nums). *Optimization:* expected_sum = n*(n+1)//2; return expected_sum - sum(nums). Or xor: res = 0; for i in range(n+1): res ^= i; for x in nums: res ^= x; return res. *Trade-off:* O(n) time, O(1) space. **O(n), O(1)**

---

### 128. Sum of Two Integers
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Bitwise: sum = a^b (no carry), carry = (a&b)<<1; repeat until carry 0. *Brute force:* Can't use +; use increment/decrement in loop — slow. *Optimization:* while b: carry = (a&b)<<1; a = a^b; b = carry. Python: mask 0xFFFFFFFF for 32-bit; handle negative. *Trade-off:* O(1) for bounded int (at most 32 iterations). **O(1), O(1)**

---

### 129. Reverse Integer
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Convert to string, reverse, back to int; or pop digits with x%10 and x//10, build result. *Brute force:* String reverse. *Optimization:* sign = 1 if x>=0 else -1; x = abs(x); res = 0; while x: res = res*10 + x%10; x //= 10. Check overflow (e.g. res > 2**31-1) return 0. *Trade-off:* O(log x) digits, O(1) space. **O(log x), O(1)**

---

## Math & Geometry

### 130. Rotate Image
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Rotate 90° = transpose then reverse each row; or rotate in place layer by layer (4-way swap). *Brute force:* New matrix B[j][n-1-i] = A[i][j]. O(n²) time, O(n²) space. *Optimization:* In-place: for layer 0 to n//2, for i in layer to n-1-layer: swap 4 corners (i,j)→(j,n-1-i)→(n-1-i,n-1-j)→(n-1-j,i). *Trade-off:* O(n²) time, O(1) space. **O(n²), O(1)**

---

### 131. Spiral Matrix
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Track top, bottom, left, right; while top<=bottom and left<=right: go right, down, left, up; shrink bounds. *Brute force:* Visit in spiral order with a direction state. *Optimization:* t,b,l,r = 0,m-1,0,n-1. While t<=b and l<=r: right (t,l→r), t++; down (r,t→b), r--; if t<=b: left (b,r→l), b--; if l<=r: up (l,b→t), l++. *Trade-off:* O(m·n) time, O(1) extra (output doesn't count). **O(m·n), O(1)**

---

### 132. Set Matrix Zeroes
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Use first row and first column as flags for "this row/col has zero"; then set zeros; handle row0/col0 separately. *Brute force:* Copy matrix, scan and set rows/cols. O(m·n) space. *Optimization:* First pass: mark row0 and col0 if any zero there. For i,j in 1..: if matrix[i][j]==0: matrix[i][0]=0, matrix[0][j]=0. Second pass: set to 0 using row0/col0. Clear row0/col0 at end if needed. *Trade-off:* O(m·n) time, O(1) space. **O(m·n), O(1)**

---

### 133. Happy Number
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Repeat: n = sum of squares of digits. If we reach 1, true; if we cycle (see a number again), false. Use set or Floyd cycle detection. *Brute force:* Loop and check for 1 or repeat. *Optimization:* def next(n): return sum(int(d)**2 for d in str(n)). slow = fast = n; while fast != 1: slow = next(slow); fast = next(next(fast)); if slow == fast return false. *Trade-off:* O(log n) time (digits), O(1) space with Floyd. **O(log n), O(1)**

---

### 134. Plus One
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Add 1 from right, carry; if all 9's we need new digit at front. *Brute force:* Convert to int, add 1, back to list. (May overflow.) *Optimization:* For i from len-1 down to 0: digits[i]+=1; if digits[i]==10: digits[i]=0; else return digits. If we exit loop, return [1]+digits. *Trade-off:* O(n) time, O(1) space (or O(n) if new array for all 9's). **O(n), O(1)**

---

### 135. Pow(x, n)
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Binary exponentiation: x^n = (x^(n/2))^2 if n even; x * x^(n-1) if odd. *Brute force:* Multiply x, n times. O(n). *Optimization:* n might be negative → x = 1/x, n = -n. res = 1; while n: if n&1: res *= x; x *= x; n //= 2. *Trade-off:* O(log n) time, O(1) space. **O(log n), O(1)**

---

### 136. Multiply Strings
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Grade-school multiplication: num1[i]*num2[j] goes to result[i+j] and result[i+j+1] (carry). *Brute force:* Convert to int, multiply, convert back (may overflow). *Optimization:* len(result) = len(num1)+len(num2). For i,j: mul = int(num1[i])*int(num2[j]); add to result[i+j+1] and carry to result[i+j]. Then handle carries. Remove leading zeros. *Trade-off:* O(n·m) time, O(n+m) space. **O(n·m), O(n+m)**

---

### 137. Detect Squares
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* add: store point in list and in Counter. count: for each point p1, try to form square with query point; need two points that form a square with (qx,qy) and p1. *Brute force:* count: for each pair of points check if they form square with query. O(n²). *Optimization:* Store points in list; for count(qx,qy), for each (x,y) in points: if (x,y) != (qx,qy) and same diagonal (abs(dx)==abs(dy)), the other two corners are (qx,y) and (x,qy); if both in Counter, add count. *Trade-off:* add O(1), count O(n), O(n) space. **O(1) add, O(n) count, O(n)**

---

## Additional (61–150 completed; 138–150 below)

### 138. Encode and Decode Strings
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Use length prefix: "4#word" so we know how many chars to read. Or escape a delimiter. *Brute force:* Join with a char that doesn't appear — fails if it does. *Optimization:* Encode: for each s, write str(len(s)) + '#' + s. Decode: read until '#', then read that many chars; repeat. *Trade-off:* O(n) encode/decode, O(1) extra. **O(n), O(1)**

---

### 139. Search a 2D Matrix
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Binary search on flattened index (mid → row=mid//n, col=mid%n); or binary search row then col. *Brute force:* Scan all. O(m·n). *Optimization:* Treat as sorted 1D array of length m*n; mid = (lo+hi)//2, cell = matrix[mid//n][mid%n]. *Trade-off:* O(log(m·n)) time, O(1) space. **O(log(m·n)), O(1)**

---

### 140. Time Based Key-Value Store
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Map key → list of (timestamp, value); get(key, ts): binary search for largest timestamp <= ts. *Brute force:* set: append (ts, val). get: linear scan for largest ts <= timestamp. O(n) get. *Optimization:* key → sorted list of (ts, val). get: bisect_right to find index, return list[index-1].value. *Trade-off:* set O(1), get O(log n) per key. **O(1) set, O(log n) get**

---

### 141. Median of Two Sorted Arrays
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Binary search on partition in smaller array: left half has (i+j) = (m+n+1)//2 elements; check maxLeft <= minRight. *Brute force:* Merge and take median. O(m+n). *Optimization:* We want partition such that all left elements <= all right. Binary search i in [0,m]; j = (m+n+1)//2 - i; check A[i-1]<=B[j] and B[j-1]<=A[i]. Median from max(left)+min(right). *Trade-off:* O(log min(m,n)) time, O(1) space. **O(log min(m,n)), O(1)**

---

### 142. Reorder List
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Find mid, reverse second half, merge two halves (first, last, first.next, last.next...). *Brute force:* Copy to array, reorder, rebuild. O(n) time, O(n) space. *Optimization:* Slow/fast to find mid; reverse second half; merge: take one from head, one from reversed head, alternate. *Trade-off:* O(n) time, O(1) space. **O(n), O(1)**

---

### 143. Remove Nth Node From End of List
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Two passes (count then remove); or one pass with two pointers (gap n+1). *Brute force:* Count length L, remove (L-n)th node. *Optimization:* Dummy; fast and slow start at dummy. Move fast n+1 times; then move both until fast is null. slow.next is the node to remove. *Trade-off:* O(n) time, O(1) space. **O(n), O(1)**

---

### 144. Copy List with Random Pointer
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Hash map old→new (like Clone Graph); or interweave copies then set random then separate. *Brute force:* Map each node to copy; second pass set next and random. O(n) time, O(n) space. *Optimization:* Interweave: create copy after each node; copy.random = old.random.next; then separate lists. O(n) time, O(1) extra. *Trade-off:* Map is simpler; interweave saves space. **O(n), O(1)**

---

### 145. Add Two Numbers
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Dummy head; while l1 or l2 or carry: sum = (l1.val if l1 else 0) + (l2.val if l2 else 0) + carry; append sum%10, carry = sum//10. *Brute force:* Convert lists to ints, add, convert back — may overflow. *Optimization:* One pass with carry; create node for each digit. *Trade-off:* O(max(m,n)) time, O(1) extra (result list doesn't count). **O(max(m,n)), O(1)**

---

### 146. Find the Duplicate Number
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Array as hash (mark visited); or Floyd cycle (value = next index, so cycle exists). *Brute force:* Set or count array. O(n) space. *Optimization:* Floyd: slow = fast = 0; do slow = nums[slow], fast = nums[nums[fast]] until meet; then slow = 0, move both 1 step until meet — that's the duplicate. *Trade-off:* O(n) time, O(1) space. **O(n), O(1)**

---

### 147. Merge K Sorted Lists
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Min-heap of k heads; or merge two at a time (1+2, then +3...). *Brute force:* Collect all, sort, build list. O(N log N). *Optimization:* Heap: push (node.val, i, node) for each list head. Pop smallest, append to result, push next from that list. *Trade-off:* O(N log k) time, O(k) heap. **O(N log k), O(k)**

---

### 148. Reverse Nodes in K-Group
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Get k nodes, reverse them, connect to prev and next group; repeat. *Brute force:* Copy to array, reverse in chunks, rebuild. *Optimization:* Count nodes; for each group of k: reverse that segment (like reverse linked list), connect group's tail to next group's head. *Trade-off:* O(n) time, O(1) space. **O(n), O(1)**

---

### 149. Car Fleet
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* Sort by position (descending); time to target = (target - pos) / speed. Cars that catch up merge into one fleet. *Brute force:* Simulate time — messy. *Optimization:* Sort (position, speed) by position descending. Stack of (time to target). For each car: t = (target - pos) / speed. While stack and t >= stack[-1]: pop (car behind catches fleet). Push t. Return len(stack). *Trade-off:* O(n log n) time, O(n) space. **O(n log n), O(n)**

---

### 150. Merge Triplets to Form Target
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________
**Complexity:** O(___) time, O(___) space

**Answer (discussion):** *Choices:* We can only take triplets where each dimension <= target. We need to reach (tx, ty, tz). If any triplet has a dimension > target, skip. *Brute force:* Try all subsets of triplets — exponential. *Optimization:* Keep (a, b, c) = max so far. For each triplet (x,y,z): if (x,y,z) <= (tx,ty,tz), update (a,b,c) = (max(a,x), max(b,y), max(c,z)). Return (a,b,c)==(tx,ty,tz). *Trade-off:* O(n) time, O(1) space. **O(n), O(1)**
