## Context

When a warp hits a divergent `if`, its lanes split into two groups that
execute different code paths. Older GPUs guaranteed that once both
groups finished, they'd automatically be back "in lockstep"
(**reconverged**) at the very next instruction after the `if` -- so code
that placed a warp-shuffle right after a divergent block, with a
full-warp mask like `0xffffffff`, was safe: every lane really was there,
together, when the shuffle ran. Newer hardware (Volta and later, with
*independent thread scheduling*) removed that automatic guarantee --
each divergent group can now be scheduled at its own pace, and there is
no promise they rejoin at any particular point unless the program says
so explicitly (with `__syncwarp()`, or here, `__syncthreads()`).

A shuffle that assumes a mask of lanes are all present *right now*, when
one of them hasn't actually reached that instruction yet, doesn't fail
loudly -- it just silently returns whatever that lane published the last
time it *did* reach a shuffle at that point, which for a lane that
simply hasn't gotten there yet is nothing meaningful at all.

## Task

`solve.cu` puts a `__syncthreads()` **inside** the `if (tid < 16)`
block, so only lanes `0..15` are synchronized there -- lanes `16..31`
skip straight past it to the shuffle. Move the synchronization so it
applies to **every** lane before any lane reaches the shuffle:

```cuda
__global__ void divergent_shuffle(float* out, const float* in, int n) {
    int tid = threadIdx.x;
    float val = in[tid];
    if (tid < 16) {
        val = val * 2.0;
    }
    // <-- the synchronization belongs here, unconditional
    float shuffled = __shfl_up_sync(0xffffffff, val, 1);
    out[tid] = shuffled;
}
```

## Example

For 32 fixed input values with lanes `0..15` doubled by the branch, the
correct `shfl_up(val, 1)` result at lane 16 is lane 15's (doubled)
value. With the synchronization stuck inside the `if`, lane 15
hasn't reached the shuffle yet when lane 16 does (lane 16 took the path
with no extra barrier and gets there first), so lane 16 reads back its
own value instead -- one specific, exactly reproducible wrong entry in
the output, at exactly the branch boundary.

## What the gate checks

`check.py` runs the kernel over 32 fixed random values (one full warp)
and checks `max_abs_err <= 1e-9` against a `numpy` oracle: lanes
`0..15` doubled, then every lane's output is its neighbor one lane
below (`shfl_up`, with lane `0` keeping its own value since there's no
lane `-1`). The reference (synchronization moved outside the `if`)
matches exactly. The buggy version matches on every lane except lane
`16`, where it returns its own value (`off by roughly the doubled
neighbor's magnitude`) instead of lane `15`'s -- one wrong entry is
enough to fail `max_abs_err`, and it's the exact lane where the
`if`'s divergence boundary sits.
