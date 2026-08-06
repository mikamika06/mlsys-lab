## Context

In a two‑dimensional Euclidean space a counter‑clockwise rotation by an angle $\theta$ is represented by the matrix
$$R(\theta)=\begin{bmatrix}\cos \theta & -\sin \theta\\\sin \theta & \;\cos \theta\end{bmatrix}.$$
Applying $R(\theta)$ to a vector $v=[x,y]^\top$ yields the rotated vector $w=R(\theta)v$.
The signed angle between two non‑zero vectors $v,w$ can be recovered from their dot and cross products:
$$\theta = \operatorname{atan2}\!\bigl(v_x w_y - v_y w_x,\; v_x w_x + v_y w_y\bigr).$$

## Task

Implement `recover_angles(orig, rot)` that takes two list of shape $(n,2)$ containing $n$ original vectors and their corresponding rotated versions. The function must return a one‑dimensional array of length $n$ with the signed rotation angles (in radians) for each pair. Use only vectorised Python operations; no explicit Python loops.

## Example

```python
import math

orig = [[1, 0], [0, 1]]
theta = math.pi / 4
rot = [[v[0] * math.cos(theta) + v[1] * math.sin(theta), -v[0] * math.sin(theta) + v[1] * math.cos(theta)]
       for v in orig]
angles = recover_angles(orig, rot)
print(angles)  # [-0.7853981633974482, -0.7853981633974482]
```

## What the gate checks

The grader computes a reference solution using the `atan2` formula above and compares it to your output with the scorer

```python
max_abs_err(reference, candidate)
```

A submission passes if the maximum absolute error is at most $10^{-5}$.
