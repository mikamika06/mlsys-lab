## Context

A slab allocator serves every allocation from one of a small, fixed set
of block sizes ("size classes") instead of an exact fit, so it never has
to deal with arbitrary-sized free gaps — at the cost of rounding every
request up to the next class. That rounding is **internal
fragmentation**: memory that is allocated but never used by the
requester. The worst case is a request 1 byte over a class boundary,
which gets bumped all the way up to the next class.

## Task

Implement

```cpp
double slab_fragmentation_ratio(const int* size_classes, int num_classes, const int* requests, int n);
```

`size_classes[0..num_classes)` is sorted ascending. For each request
`requests[i]`, find the smallest size class `>= requests[i]` and compute
`allocated / requests[i]`. Return the average of these `n` ratios.

## Example

With classes `16, 32, 64, 128, 256`: a 1-byte request rounds up to `16`
(ratio `16.0`, the worst case here); a 17-byte request rounds up to `32`
(ratio `≈1.882`); a 32-byte or 256-byte request is an exact fit (ratio
`1.0`). Averaging all 6 requests in the fixture gives `≈3.858` — pulled
way up by that one pathological 1-byte request, a common shape for this
kind of average.

## What the gate checks

`max_abs_err` on the printed average ratio for 6 fixed requests.
Rounding down instead of up, using the wrong size class (e.g. the first
one `>` instead of `>=` the request), or averaging the allocated sizes
instead of the ratios all give a different number; a starter returning
`0.0` fails outright.
