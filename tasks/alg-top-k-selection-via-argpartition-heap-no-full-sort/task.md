## Context
In many machine learning algorithms like k-Nearest Neighbors (kNN), we need to find the $k$ closest points (i.e., the $k$ smallest distances). A naive implementation might sort the entire array of distances, taking $O(N \log N)$ time. However, this is inefficient when $N$ is large and $k$ is small. 

A better approach is to use a max-heap of size $k$, or the Quickselect algorithm (which `numpy.argpartition` uses), to find the top $k$ elements in $O(N \log k)$ or $O(N)$ time.

## Task
Implement the function `k_smallest_indices(arr, k)` that returns the indices of the `k` smallest elements in the list `arr`. The returned list of indices does not need to be sorted.

You are NOT allowed to use Python's built-in `sort()` or `sorted()` functions. 

## Example
```python
arr = [9.0, 1.0, 4.0, 7.0, 2.0, 5.0]
k = 3
k_smallest_indices(arr, k)
# Could return [1, 4, 2] since arr[1]=1.0, arr[4]=2.0, arr[2]=4.0 are the 3 smallest.
```

## What the gate checks
- `correct`: Your function must return exactly the same set of indices as the optimal solution for various inputs.
- `used_sort`: We will trace your function's execution to ensure `list.sort` or `sorted` are NOT called.
