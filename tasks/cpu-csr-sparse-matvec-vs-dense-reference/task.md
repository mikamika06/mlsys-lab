## Context

In a *Compressed Sparse Row* (CSR) matrix $A \in \mathbb{R}^{n\times d}$ the non‑zero elements are stored in three parallel arrays:

$$\begin{aligned}
\texttt{data}   &:& A_{\text{nz}} \in \mathbb{R}^{m} \\
\texttt{indices}&:& C_{\text{nz}} \in \{0,\dots,d-1\}^{m} \\
\texttt{indptr} &::& R \in \{0,\dots,m\}^{n+1}
\end{aligned}$$

For row $i$ the non‑zero values lie in
$\texttt{data}[\,R_i : R_{i+1}\,]$ and their column indices are
$\texttt{indices}[\,R_i : R_{i+1}\,]$.  
The matrix–vector product $y = A\,x$, where $x \in \mathbb{R}^{d}$, can be computed by summing the contributions of each row.

## Task

Implement a function that performs this multiplication efficiently:

```python
def csr_matvec(data: list[float], indices: list[int], indptr: list[int], x: list[float]) -> list[float]:
    ...
```

The function receives the CSR representation of a matrix and a dense vector `x`.  
It must return the product `y` as a 1‑D list of type `float64`.

* Constraints
  * Use only Python; do not convert the sparse matrix to a dense form.
  * The algorithm should run in $O(\text{nnz})$ time, where $\text{nnz}=m$ is the number of stored non‑zeros.

## Example

```python

# A = [[1, 0, 2],
#      [0, 3, 0]]
data   = [1, 2, 3]
indices= [0, 2, 1]
indptr = [0, 2, 3]

x = [4, 5, 6]

y = csr_matvec(data, indices, indptr, x)
print(y)  # [16.0, 15.0]
```

## What the gate checks

The returned vector is compared against a dense reference implementation using the scorer
`max_abs_err`.  
The candidate passes if

$$\displaystyle \max_{i} |y_i - y^{\text{ref}}_i| \;\leq\; 10^{-9}\,.$$

The test harness generates several random CSR matrices and vectors at grading time.
