## Context

The Euclidean distance between two points $a, b \in \mathbb{R}^d$ is

$$\lVert a - b\rVert^2 = \sum_{i=1}^{d}(a_i-b_i)^2.$$

For a dataset of $n$ points stored as rows in a list $X \in \mathbb{R}^{n\times d}$, the brute‑force $k$‑nearest‑neighbour (kNN) algorithm evaluates this distance for every pair $(q,p)$ where $q$ is a query point and $p$ ranges over all other points.  The total number of distance evaluations performed by the naive approach is therefore

$$\text{brute}_{\text{count}} = n(n-1).$$

A kd‑tree partitions space recursively along alternating axes, storing one point per node.  During a search it first descends to the leaf that contains the query, then backtracks while pruning subtrees whose bounding hyperplanes are farther than the current $k$‑th best distance.  Because many nodes can be skipped, the number of distance evaluations in practice is far smaller:

$$\text{kd}_{\text{count}} \ll n(n-1).$$

Counting these evaluations gives a clean, implementation‑independent metric for algorithmic efficiency.

## Task

Implement the following function:

```python
def count_distance_computations(points: list[list[float]], k: int) -> tuple[int, int]:
    """
    Return (brute_count, kd_count).

    * `points` is an (n, d) array of real numbers.
    * `k` is a positive integer < n.

    The function must count each Euclidean distance evaluation between
    a query point and a data point exactly once per evaluation,
    excluding self‑distances.  It should use only Python; no external
    libraries are allowed.
    """
```

The returned tuple must contain the exact counts for the brute‑force
and kd‑tree algorithms as described above.

## Example

```python
points = [[0, 0], [1, 0], [0, 2], [3, 3]]
k = 1
brute_count, kd_count = count_distance_computations(points, k)
print(brute_count)   # 12  (4 * 3)
print(kd_count)      # e.g. 8  (depends on the tree layout)
```

## What the gate checks

The grader computes a reference pair of counts using a correct kd‑tree
implementation and compares it to the candidate’s output.
Both numbers must match exactly; otherwise the submission fails.
Because the kd‑tree algorithm prunes many nodes, its count will be
strictly smaller than the brute‑force count.  A naive implementation
that reports `kd_count == brute_count` will not pass.
