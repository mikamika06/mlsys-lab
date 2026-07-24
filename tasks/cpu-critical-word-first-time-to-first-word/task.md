## Context

After a cache miss, DRAM doesn't hand back one word instantly — it streams
the whole line as a **burst**: the first word lands after some fixed
`base_latency` (row activation, command overhead, ...), and every
subsequent word of the line follows one every `cycles_per_word` cycles.
The CPU is stalled until the *specific* word it actually asked for — the
**critical word** — comes back, not until the whole line finishes.

A naive memory controller always bursts the line starting at word 0, in
address order. If the CPU happened to miss on word 6 of an 8-word line, it
has to sit through words 0–5 first even though it doesn't care about any
of them. **Critical-word-first (CWF)**, plus **early restart**, fixes this:
the controller starts the burst *at the requested word* and wraps around
to fill the rest afterward — the CPU's critical word is always the very
first thing to arrive.

## Task

Implement

```cpp
long time_to_word(int line_words, int start_word, int target_word,
                   int base_latency, int cycles_per_word);
```

modeling one burst of `line_words` words that starts at `start_word` and
wraps circularly (`start_word, start_word+1, ..., line_words-1, 0, 1, ...`)
until every word has been transferred once. Return the cycle at which
`target_word` becomes available:

$$
\text{position} = ((\text{target\_word} - \text{start\_word}) \bmod \text{line\_words} + \text{line\_words}) \bmod \text{line\_words}
$$
$$
\text{time} = \text{base\_latency} + \text{position} \times \text{cycles\_per\_word}
$$

`position` must land in $[0, \text{line\_words})$ even when
`target_word < start_word` — C++'s `%` returns a **negative** result for a
negative left-hand side, so a raw `(target_word - start_word) % line_words`
is not enough on its own.

Without CWF the burst starts at word 0 (`start_word = 0`); with CWF it
starts at the requested word itself (`start_word = target_word`) — the
driver calls this same function both ways per scenario.

## Example

`line_words=8, base_latency=40, cycles_per_word=4`, requested word `6`:
without CWF, `position = (6 - 0) mod 8 = 6`, so `time = 40 + 6*4 = 64`.
With CWF, `start_word = target_word = 6`, so `position = (6-6) mod 8 = 0`
and `time = 40` — the critical word arrives 24 cycles sooner, with no
change to how long the *whole line* takes to finish.

## What the gate checks

`main.cpp` sweeps every possible requested word across 4 fixed
`(line_words, base_latency, cycles_per_word)` memory-system configurations
(covering different line widths and burst speeds) and prints, for each,
the time-to-first-useful-word without CWF, with CWF, and the cycles
saved — 36 lines total. The candidate's full stdout is compared
byte-for-byte (`exact_match = 1.0`) against the reference's. Forgetting
the wraparound (using `target_word` directly as the position instead of
`target_word - start_word`), or leaving the negative-modulo case
unhandled, produces numbers that are only right for a subset of the
fixture's `(start_word, target_word)` pairs — most rows still mismatch.
