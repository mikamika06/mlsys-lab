## Context

Hadamard matrices are useful as fast orthogonal mixing transforms. A Hadamard matrix $H$ with entries in $\{-1,1\}$ satisfies

$$
H H^\top = n I
$$

for a matrix size $n$. This means the raw Hadamard transform is not an orthogonal rotation because it changes vector lengths by a factor of $\sqrt{n}$.

A production rotation uses the normalized matrix

$$
Q = \frac{1}{\sqrt{n}} H ,
$$

which gives

$$
Q Q^\top = I .
$$

For matrices $X$ and $W$, a rotation should preserve the product:

$$
(XQ)(Q^\top W) = X(QQ^\top)W = XW .
$$

If the normalization is missing, the result becomes

$$
(XH)(H^\top W) = nXW ,
$$

which breaks the invariance property.

## Task

Implement `hadamard_rotate(X, W)`:

```python
def hadamard_rotate(X, W):
    ...
```

The function receives a matrix $X \in \mathbb{R}^{m \times n}$ and a matrix $W \in \mathbb{R}^{n \times k}$, where $n$ is a power of two. Return two matrices `(X_rot, W_rot)` such that:

$$
X_{\text{rot}} = XQ
$$

and

$$
W_{\text{rot}} = Q^\top W
$$

where $Q$ is the normalized Hadamard matrix. Use Python operations only. The returned values must be `float64`.

## Example

```python

X = [[1.0, 2.0], [3.0, 4.0]]
W = [[2.0], [1.0]]

X_rot, W_rot = hadamard_rotate(X, W)

# X_rot @ W_rot is equal to X @ W
```

## What the gate checks

The gate constructs the real Hadamard oracle, normalizes it by $1/\sqrt{n}$, and checks that the returned rotated matrices preserve the original product. The maximum absolute error

$$
\max_i |A_i - B_i|
$$

between the oracle product and the submitted product must be less than $10^{-5}$. A solution using the unnormalized Hadamard matrix fails because the product is scaled by $n$ instead of being invariant.
