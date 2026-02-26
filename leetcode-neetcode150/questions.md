# NeetCode 150 – Questions with Examples

A curated list of 150 frequently asked LeetCode problems for interview preparation. Each entry includes the problem statement and example.

---

## Arrays & Hashing

### 1. Contains Duplicate (LC 217) – Easy
Given an integer array `nums`, return `true` if any value appears at least twice in the array.

**Example:**
```
Input: nums = [1,2,3,1]
Output: true
```

### 2. Valid Anagram (LC 242) – Easy
Given two strings `s` and `t`, return `true` if `t` is an anagram of `s`.

**Example:**
```
Input: s = "anagram", t = "nagaram"
Output: true
```

### 3. Two Sum (LC 1) – Easy
Given an array of integers `nums` and an integer `target`, return indices of the two numbers that add up to `target`.

**Example:**
```
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
```

### 4. Group Anagrams (LC 49) – Medium
Given an array of strings `strs`, group the anagrams together.

**Example:**
```
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
```

### 5. Top K Frequent Elements (LC 347) – Medium
Given an integer array `nums` and an integer `k`, return the `k` most frequent elements.

**Example:**
```
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
```

### 6. Product of Array Except Self (LC 238) – Medium
Given an integer array `nums`, return an array `answer` such that `answer[i]` equals the product of all elements of `nums` except `nums[i]`.

**Example:**
```
Input: nums = [1,2,3,4]
Output: [24,12,8,6]
```

### 7. Valid Sudoku (LC 36) – Medium
Determine if a 9×9 Sudoku board is valid.

**Example:**
```
Input: board = [["5","3",".",...], ...]
Output: true
```

### 8. Encode and Decode Strings (LC 271) – Medium
Design an algorithm to encode a list of strings to a string and decode a string to a list of strings.

**Example:**
```
Input: ["Hello","World"]
Encoded: "5#Hello5#World"
```

### 9. Longest Consecutive Sequence (LC 128) – Medium
Given an unsorted array of integers `nums`, return the length of the longest consecutive elements sequence.

**Example:**
```
Input: nums = [100,4,200,1,3,2]
Output: 4  (sequence [1,2,3,4])
```

---

## Two Pointers

### 10. Valid Palindrome (LC 125) – Easy
Given a string `s`, determine if it is a palindrome (considering only alphanumeric chars, ignore case).

**Example:**
```
Input: s = "A man, a plan, a canal: Panama"
Output: true
```

### 11. Two Sum II – Input Array Is Sorted (LC 167) – Medium
Given a 1-indexed sorted array and target, find two numbers that add up to target.

**Example:**
```
Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
```

### 12. 3Sum (LC 15) – Medium
Given an integer array `nums`, return all unique triplets that sum to 0.

**Example:**
```
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
```

### 13. Container With Most Water (LC 11) – Medium
Find two lines that form a container with the most water.

**Example:**
```
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
```

### 14. Trapping Rain Water (LC 42) – Hard
Given `n` non-negative integers representing an elevation map, compute how much water it can trap.

**Example:**
```
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
```

---

## Sliding Window

### 15. Best Time to Buy and Sell Stock (LC 121) – Easy
You can only complete one transaction. Find the maximum profit.

**Example:**
```
Input: prices = [7,1,5,3,6,4]
Output: 5  (buy 1, sell 6)
```

### 16. Longest Substring Without Repeating Characters (LC 3) – Medium
Find the length of the longest substring without repeating characters.

**Example:**
```
Input: s = "abcabcbb"
Output: 3  ("abc")
```

### 17. Longest Repeating Character Replacement (LC 424) – Medium
You can replace at most `k` characters. Find the length of the longest substring containing the same letter.

**Example:**
```
Input: s = "AABABBA", k = 1
Output: 4  ("AAAA" or "BBBB")
```

### 18. Permutation in String (LC 567) – Medium
Given two strings `s1` and `s2`, return `true` if `s2` contains a permutation of `s1`.

**Example:**
```
Input: s1 = "ab", s2 = "eidbaooo"
Output: true  (permutation "ba" in s2)
```

### 19. Minimum Window Substring (LC 76) – Hard
Given strings `s` and `t`, return the minimum window substring of `s` that contains every character in `t`.

**Example:**
```
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
```

### 20. Sliding Window Maximum (LC 239) – Hard
Given an array `nums` and integer `k`, return max element in each sliding window of size `k`.

**Example:**
```
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
```

---

## Stack

### 21. Valid Parentheses (LC 20) – Easy
Given a string `s` containing `()`, `[]`, `{}`, determine if the input is valid.

**Example:**
```
Input: s = "()[]{}"
Output: true
```

### 22. Min Stack (LC 155) – Medium
Design a stack that supports push, pop, top, and retrieving the minimum element in O(1) time.

**Example:**
```
push(-2), push(0), push(-3)
getMin() → -3
pop(), top() → 0
getMin() → -2
```

### 23. Evaluate Reverse Polish Notation (LC 150) – Medium
Evaluate an expression in Reverse Polish Notation.

**Example:**
```
Input: tokens = ["2","1","+","3","*"]
Output: 9  (2+1=3, 3*3=9)
```

### 24. Generate Parentheses (LC 22) – Medium
Given `n` pairs of parentheses, generate all combinations of well-formed parentheses.

**Example:**
```
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]
```

### 25. Daily Temperatures (LC 739) – Medium
Given daily temperatures `temperatures`, return an array such that `answer[i]` is days to wait for warmer.

**Example:**
```
Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]
```

### 26. Car Fleet (LC 853) – Medium
There are `n` cars at given positions and speeds. How many fleets arrive at the target?

**Example:**
```
Input: target = 12, position = [10,8,0,5,3], speed = [2,4,1,1,3]
Output: 3
```

### 27. Largest Rectangle in Histogram (LC 84) – Hard
Given heights of bars, find the largest rectangle area.

**Example:**
```
Input: heights = [2,1,5,6,2,3]
Output: 10
```

---

## Binary Search

### 28. Binary Search (LC 704) – Easy
Given sorted array and target, return index or -1.

**Example:**
```
Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
```

### 29. Search a 2D Matrix (LC 74) – Medium
Integers in each row are sorted; first integer of each row > last integer of previous. Search for target.

**Example:**
```
Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
Output: true
```

### 30. Koko Eating Bananas (LC 875) – Medium
Piles of bananas, guard must finish in `h` hours. Find minimum eating speed `k`.

**Example:**
```
Input: piles = [3,6,7,11], h = 8
Output: 4
```

### 31. Find Minimum in Rotated Sorted Array (LC 153) – Medium
Sorted array rotated at unknown pivot. Find minimum.

**Example:**
```
Input: nums = [3,4,5,1,2]
Output: 1
```

### 32. Search in Rotated Sorted Array (LC 33) – Medium
Search target in rotated sorted array.

**Example:**
```
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
```

### 33. Time Based Key-Value Store (LC 981) – Medium
Design a time-based key-value store that supports setting and getting with timestamps.

### 34. Median of Two Sorted Arrays (LC 4) – Hard
Find median of two sorted arrays in O(log(m+n)).

**Example:**
```
Input: nums1 = [1,3], nums2 = [2]
Output: 2.0
```

---

## Linked List

### 35. Reverse Linked List (LC 206) – Easy
Reverse a singly linked list.

**Example:**
```
Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]
```

### 36. Merge Two Sorted Lists (LC 21) – Easy
Merge two sorted linked lists.

**Example:**
```
Input: l1 = [1,2,4], l2 = [1,3,4]
Output: [1,1,2,3,4,4]
```

### 37. Reorder List (LC 143) – Medium
Reorder: L0 → Ln → L1 → Ln-1 → ...

### 38. Remove Nth Node From End of List (LC 19) – Medium
Remove the nth node from the end.

**Example:**
```
Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]
```

### 39. Copy List with Random Pointer (LC 138) – Medium
Deep copy a linked list with a random pointer.

### 40. Add Two Numbers (LC 2) – Medium
Two linked lists represent digits in reverse. Add them.

**Example:**
```
Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]  (342 + 465 = 807)
```

### 41. Linked List Cycle (LC 141) – Easy
Detect if linked list has a cycle.

### 42. Find the Duplicate Number (LC 287) – Medium
Array of n+1 integers in [1,n]; find the duplicate (Floyd's cycle detection).

### 43. LRU Cache (LC 146) – Medium
Design LRU (Least Recently Used) cache with get and put in O(1).

### 44. Merge K Sorted Lists (LC 23) – Hard
Merge k sorted linked lists.

### 45. Reverse Nodes in K-Group (LC 25) – Hard
Reverse nodes k at a time.

---

## Trees

### 46. Invert Binary Tree (LC 226) – Easy
Invert the tree (swap left/right children).

### 47. Maximum Depth of Binary Tree (LC 104) – Easy
Return the maximum depth.

**Example:**
```
Input: [3,9,20,null,null,15,7]
Output: 3
```

### 48. Diameter of Binary Tree (LC 543) – Easy
Longest path between any two nodes (may not pass root).

### 49. Balanced Binary Tree (LC 110) – Easy
Check if tree is height-balanced (|left - right| ≤ 1).

### 50. Same Tree (LC 100) – Easy
Check if two trees are structurally identical.

### 51. Subtree of Another Tree (LC 572) – Easy
Check if `subRoot` is a subtree of `root`.

### 52. Lowest Common Ancestor of BST (LC 235) – Medium
Find LCA in a BST.

### 53. Binary Tree Level Order Traversal (LC 102) – Medium
Return level-by-level traversal.

### 54. Binary Tree Right Side View (LC 199) – Medium
Return values of rightmost nodes at each level.

### 55. Count Good Nodes in Binary Tree (LC 1448) – Medium
Count nodes where value ≥ all ancestors.

### 56. Validate Binary Search Tree (LC 98) – Medium
Check if tree is valid BST.

### 57. Kth Smallest Element in BST (LC 230) – Medium
Return kth smallest element (1-indexed).

### 58. Construct Binary Tree from Preorder and Inorder (LC 105) – Medium
Build tree from preorder and inorder.

### 59. Binary Tree Maximum Path Sum (LC 124) – Hard
Max path sum (any node to any node).

### 60. Serialize and Deserialize Binary Tree (LC 297) – Hard
Serialize and deserialize a binary tree.

---

## Tries

### 61. Implement Trie (Prefix Tree) (LC 208) – Medium
Implement insert, search, startsWith.

### 62. Design Add and Search Words Data Structure (LC 211) – Medium
Design WordDictionary with add and search ('.' matches any letter).

### 63. Word Search II (LC 212) – Hard
Given board and list of words, return all words on the board.

---

## Heap / Priority Queue

### 64. Kth Largest Element in a Stream (LC 703) – Easy
Design a class to find kth largest in a stream.

### 65. Last Stone Weight (LC 1046) – Easy
Smash two heaviest stones; return last remaining weight.

### 66. K Closest Points to Origin (LC 973) – Medium
Return k closest points to origin.

### 67. Kth Largest Element in an Array (LC 215) – Medium
Find kth largest element (QuickSelect or heap).

### 68. Task Scheduler (LC 621) – Medium
Same task needs n units cooldown. Find minimum time.

### 69. Design Twitter (LC 355) – Medium
Post tweet, get news feed, follow/unfollow.

### 70. Find Median from Data Stream (LC 295) – Hard
Support addNum and findMedian.

---

## Backtracking

### 71. Subsets (LC 78) – Medium
Return all subsets (power set).

**Example:**
```
Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

### 72. Combination Sum (LC 39) – Medium
Find all combinations that sum to target (reuse allowed).

### 73. Permutations (LC 46) – Medium
Return all permutations.

### 74. Subsets II (LC 90) – Medium
All subsets with duplicates (no duplicates in result).

### 75. Combination Sum II (LC 40) – Medium
Combinations that sum to target (no reuse).

### 76. Word Search (LC 79) – Medium
Search word in 2D board (adjacent cells).

### 77. Palindrome Partitioning (LC 131) – Medium
Partition string so every substring is palindrome.

### 78. Letter Combinations of a Phone Number (LC 17) – Medium
Given digits, return all letter combinations.

### 79. N-Queens (LC 51) – Hard
Place n queens on n×n board so no two attack.

### 80. N-Queens II (LC 52) – Hard
Return number of distinct N-Queens solutions.

---

## Graphs

### 81. Number of Islands (LC 200) – Medium
Count number of islands (connected 1's).

**Example:**
```
Input: grid = [["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]]
Output: 3
```

### 82. Clone Graph (LC 133) – Medium
Deep clone an undirected graph.

### 83. Pacific Atlantic Water Flow (LC 417) – Medium
Find cells that can flow to both Pacific and Atlantic.

### 84. Surrounded Regions (LC 130) – Medium
Flip 'O' to 'X' if surrounded by 'X'.

### 85. Rotting Oranges (LC 994) – Medium
BFS: minutes until all oranges rotten.

### 86. Course Schedule (LC 207) – Medium
Can you finish all courses? (cycle detection)

### 87. Course Schedule II (LC 210) – Medium
Return valid order to finish all courses.

### 88. Redundant Connection (LC 684) – Medium
Find edge that creates cycle in tree.

### 89. Number of Connected Components (LC 323) – Medium
Count connected components in undirected graph.

### 90. Graph Valid Tree (LC 261) – Medium
Is the graph a valid tree? (n-1 edges, no cycles)

### 91. Word Ladder (LC 127) – Hard
Shortest transformation sequence from beginWord to endWord.

### 92. Word Ladder II (LC 126) – Hard
Return all shortest transformation sequences.

---

## Advanced Graphs

### 93. Reconstruct Itinerary (LC 332) – Hard
Given tickets, return lexicographically smallest itinerary.

### 94. Min Cost to Connect All Points (LC 1584) – Medium
Prim's/Kruskal's: minimum cost to connect all points.

### 95. Network Delay Time (LC 743) – Medium
Dijkstra: time for signal to reach all nodes.

### 96. Swim in Rising Water (LC 778) – Hard
Binary search + DFS: min time to reach bottom-right.

### 97. Alien Dictionary (LC 269) – Hard
Derive order of letters from sorted words.

### 98. Cheapest Flights Within K Stops (LC 787) – Medium
Bellman-Ford: cheapest path with at most k stops.

---

## 1-D Dynamic Programming

### 99. Climbing Stairs (LC 70) – Easy
Ways to climb n stairs (1 or 2 steps).

### 100. Min Cost Climbing Stairs (LC 746) – Easy
Min cost to reach top (from cost array).

### 101. House Robber (LC 198) – Medium
Max money without robbing adjacent houses.

### 102. House Robber II (LC 213) – Medium
Same but houses are arranged in a circle.

### 103. Longest Palindromic Substring (LC 5) – Medium
Return longest palindromic substring.

### 104. Palindromic Substrings (LC 647) – Medium
Count all palindromic substrings.

### 105. Decode Ways (LC 91) – Medium
Ways to decode '1'='A', '2'='B', ... '26'='Z'.

### 106. Coin Change (LC 322) – Medium
Fewest coins to make amount.

### 107. Maximum Product Subarray (LC 152) – Medium
Max product of contiguous subarray.

### 108. Word Break (LC 139) – Medium
Can string be segmented into dictionary words?

### 109. Longest Increasing Subsequence (LC 300) – Medium
Length of longest strictly increasing subsequence.

### 110. Partition Equal Subset Sum (LC 416) – Medium
Can array be partitioned into two equal-sum subsets?

---

## Intervals

### 111. Insert Interval (LC 57) – Medium
Insert new interval into non-overlapping intervals.

### 112. Merge Intervals (LC 56) – Medium
Merge all overlapping intervals.

**Example:**
```
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
```

### 113. Non-overlapping Intervals (LC 435) – Medium
Min intervals to remove for non-overlapping.

### 114. Meeting Rooms (LC 252) – Easy
Can one person attend all meetings?

### 115. Meeting Rooms II (LC 253) – Medium
Min meeting rooms needed.

### 116. Minimum Interval to Include Each Query (LC 1851) – Hard
For each query, find smallest interval containing it.

---

## Greedy

### 117. Maximum Subarray (LC 53) – Medium
Max sum of contiguous subarray (Kadane's).

**Example:**
```
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6  (subarray [4,-1,2,1])
```

### 118. Jump Game (LC 55) – Medium
Can you reach last index? (nums[i] = max jump from i)

### 119. Jump Game II (LC 45) – Medium
Min jumps to reach last index.

### 120. Gas Station (LC 134) – Medium
Can you complete circuit? gas[i], cost[i].

### 121. Hand of Straights (LC 846) – Medium
Can you partition hand into groups of size groupSize in consecutive order?

### 122. Merge Triplets to Form Target (LC 1899) – Medium
Can triplets be merged to form target?

### 123. Partition Labels (LC 763) – Medium
Partition so each letter appears in at most one part.

### 124. Valid Parenthesis String (LC 678) – Medium
'(' ')' '*' where '*' can be '(' ')' or empty. Valid?

---

## 2-D Dynamic Programming

### 125. Unique Paths (LC 62) – Medium
Robot at top-left, move right/down to bottom-right. Count paths.

### 126. Longest Common Subsequence (LC 1143) – Medium
Length of LCS of two strings.

### 127. Best Time to Buy and Sell Stock with Cooldown (LC 309) – Medium
Cooldown 1 day after sell.

### 128. Coin Change II (LC 518) – Medium
Number of combinations that make amount.

### 129. Target Sum (LC 494) – Medium
Ways to assign +/- to get target.

### 130. Interleaving String (LC 97) – Medium
Is s3 interleaving of s1 and s2?

### 131. Longest Increasing Path in a Matrix (LC 329) – Hard
Longest strictly increasing path.

### 132. Distinct Subsequences (LC 115) – Hard
Number of distinct subsequences of s equal to t.

### 133. Edit Distance (LC 72) – Medium
Min operations (insert/delete/replace) to convert word1 to word2.

### 134. Burst Balloons (LC 312) – Hard
Max coins from bursting balloons.

### 135. Regular Expression Matching (LC 10) – Hard
Match pattern '.' and '*' (zero or more of preceding).

---

## Bit Manipulation

### 136. Single Number (LC 136) – Easy
Every element appears twice except one. Find it (XOR).

### 137. Number of 1 Bits (LC 191) – Easy
Count set bits (Hamming weight).

### 138. Counting Bits (LC 338) – Easy
Return array where ans[i] = number of 1's in binary of i.

### 139. Reverse Bits (LC 190) – Easy
Reverse bits of 32-bit unsigned integer.

### 140. Missing Number (LC 268) – Easy
Array [0,n] with one missing. Find it.

### 141. Sum of Two Integers (LC 371) – Medium
Add two integers without + or -.

### 142. Reverse Integer (LC 7) – Medium
Reverse digits; handle overflow.

---

## Math & Geometry

### 143. Rotate Image (LC 48) – Medium
Rotate n×n matrix 90 degrees clockwise in-place.

### 144. Spiral Matrix (LC 54) – Medium
Return elements in spiral order.

### 145. Set Matrix Zeroes (LC 73) – Medium
If element is 0, set entire row and column to 0 (in-place).

### 146. Happy Number (LC 202) – Easy
Repeat: sum of squares of digits. Ends at 1?

### 147. Plus One (LC 66) – Easy
Increment number represented as digit array.

### 148. Pow(x, n) (LC 50) – Medium
Compute x^n (fast exponentiation).

### 149. Multiply Strings (LC 43) – Medium
Multiply two numbers as strings.

### 150. Detect Squares (LC 2013) – Medium
Design: add points, count squares formed by points.
