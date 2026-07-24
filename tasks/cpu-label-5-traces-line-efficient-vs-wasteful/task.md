## Context

A cache doesn't fetch bytes, it fetches *lines*: every access pulls in
the whole `line_bytes`-byte line its address falls in, whether or not the
rest of that line ever gets touched. A tight stride-1 sweep uses nearly
every byte of every line it fetches. A large-stride walk (say, one 4-byte
float every 64 bytes) fetches a full line for each access but only ever
uses 4 of its 64 bytes — 94% of the memory traffic is wasted. This ratio
of *useful bytes consumed* to *bytes actually pulled from memory* is a
direct measure of spatial locality.

Reuse matters too, in the other direction: re-reading data that's
already cache-resident costs no extra fetch, so a trace that revisits a
tiny working set many times can rack up far more "useful bytes consumed"
than bytes it ever fetched — a ratio well above 1.

## Task

Implement

```cpp
int classify_trace(const long* addrs, int num_accesses, int elem_bytes, int line_bytes);
```

`addrs[0..num_accesses)` is a trace of byte addresses (may repeat), each
access reading `elem_bytes` bytes starting there. Compute:

$$
\text{bytes\_used} = \text{num\_accesses} \times \text{elem\_bytes}
$$

(every access counts, including repeats — a re-read is still useful
work), and

$$
\text{bytes\_fetched} = |\{\lfloor \text{addr} / \text{line\_bytes} \rfloor : \text{addr} \in \text{addrs}\}| \times \text{line\_bytes}
$$

(the number of *distinct* lines the trace touches, times the line size).
Then

$$
\text{efficiency} = \frac{\text{bytes\_used}}{\text{bytes\_fetched}}
$$

Return `1` ("line-efficient") if `efficiency >= 0.5`, else `0`
("wasteful").

## Example

A trace of 64 accesses at stride 64 bytes, `elem_bytes = 4`,
`line_bytes = 64`: each access lands on a fresh line, so
`bytes_fetched = 64 * 64 = 4096`, but `bytes_used = 64 * 4 = 256`.
`efficiency = 256 / 4096 = 0.0625 < 0.5` → wasteful (label `0`).

## What the gate checks

The driver classifies 5 fixed traces: a contiguous stride-4 sweep
(everything used, ratio `1.0`), a stride-64 walk (ratio `0.0625`), a
stride-16 walk (ratio `0.25`), a stride-8 walk (exactly `0.5` — the
boundary case), and 200 accesses cycling through 4 addresses that all sit
in one line (heavy reuse, ratio `12.5`). It prints all 5 labels. The
grader compiles `solve.cpp` with `clang++ -O2 -std=c++20`, runs it, and
requires

$$
\mathrm{exact\_match} = 1 \iff \text{all 5 printed labels match the reference}
$$

The reference prints `labels=1,0,0,1,1`. A classifier that counts
*distinct* addresses instead of every access (ignoring reuse) mislabels
the last trace; one that uses `>` instead of `>=` mislabels the stride-8
boundary case.
