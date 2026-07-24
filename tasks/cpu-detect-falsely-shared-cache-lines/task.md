## Context

False sharing happens when two threads write to *different* variables
that happen to live in the same cache line. Even though there is no real
data race, the cache-coherence protocol still has to bounce that line
between cores on every write, killing performance as if it were one
contended variable. It must be distinguished from **true sharing**, where
multiple threads legitimately write the exact same address (a real
dependency, not a layout bug).

## Task

Implement

```cpp
int find_falsely_shared_lines(const long* addrs, const int* thread_id, int n, int line_bytes, long* out);
```

Given `n` writes (`addrs[i]` performed by `thread_id[i]`), group them by
cache line, `line = addrs[i] / line_bytes`. A line is falsely shared iff:

- its writes come from **>= 2 distinct thread ids**, AND
- those writes touch **>= 2 distinct addresses** (ruling out true sharing,
  where every thread writes the same one address).

Write the falsely-shared line ids, sorted ascending with no duplicates,
into `out` and return how many were written.

## Example

With `line_bytes = 64`: thread 0 writing byte 4 and thread 1 writing byte
60 both land in line 0 (bytes 0–63) at two different addresses — falsely
shared. Thread 0 writing byte 128 and thread 1 also writing byte 128 (line
2) is true sharing — excluded, since there is only one distinct address.
A line touched by only one thread is never falsely shared, no matter how
many times that thread writes to it.

## What the gate checks

`exact_match`: the driver prints the count and the full sorted list of
falsely-shared line ids for a fixed 10-write, 3-thread trace containing
one clear false-sharing case, one true-sharing case, and two single-
thread lines. Missing the true-sharing exclusion, missing the
distinct-address requirement, or an unsorted/duplicated result all change
the printed line, and fail the match; a starter returning `0` fails
outright since the reference finds 2 falsely-shared lines.
