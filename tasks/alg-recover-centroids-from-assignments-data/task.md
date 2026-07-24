## Context
In $k$-Means clustering, the dataset is partitioned into $k$ clusters, where each data point is assigned a cluster label. The centroid of each cluster is defined as the mean of all points assigned to that cluster. 

Given a matrix of points $X$ and a vector of cluster labels $y$, the centroid $\mathbf{c}_i$ for cluster $i$ is calculated as:
$$ \mathbf{c}_i = \frac{1}{|S_i|} \sum_{\mathbf{x} \in S_i} \mathbf{x} $$
where $S_i$ is the set of points assigned to cluster $i$.

## Task
Implement a function `recover_centroids(X, labels)` that calculates the centroids for each cluster. 
The input `X` is an $N \times D$ NumPy array representing $N$ data points of dimension $D$.
The input `labels` is a 1D array of length $N$ containing integer labels $0, 1, \dots, k-1$.
The function should return a $k \times D$ array where the $i$-th row is the centroid of cluster $i$.

## Example
If `X` has shape $(100, 2)$ and `labels` consists of integers in $\{0, 1, 2\}$, the function should return a $(3, 2)$ matrix of the three cluster centroids.

## What the gate checks
The gate checks `max_abs_err`, the maximum absolute error between the output centroids and the true reference means.
