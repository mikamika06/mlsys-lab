## Context

Removing elements from a `std::vector<T>` while iterating over it is a classic trap: `vector::erase(it)` shifts every later element down by one, which silently invalidates whatever index or iterator you were using to track your position. A naive loop that erases and then advances its index anyway ends up **skipping** the element that just shifted into the slot it erased.

The standard fix is the **erase-remove idiom**:

```cpp
v.erase(std::remove_if(v.begin(), v.end(), pred), v.end());
```

`std::remove_if` does a single forward pass, compacting every element that does *not* satisfy `pred` to the front, and returns an iterator to the new logical end. `erase` then drops everything from there to the real end in one shot -- no shifting-while-you-look bug possible.

## Task

Fix `eraseByKey` in `solve.cpp` so it removes every `Record` in `v` whose `key` equals `target`, using the erase-remove idiom, and preserves the relative order of every retained element.

The shipped version erases inside a hand-rolled index loop instead, which -- as described above -- skips every other element in a run of consecutive matches.

## Example

```cpp
std::vector<Record> v = {{1, 1.1}, {2, 2.2}, {2, 2.3}, {2, 2.4}, {3, 3.3}};
eraseByKey(v, 2);
// correct: {(1,1.1), (3,3.3)}             -- all three 2's removed
// buggy loop: {(1,1.1), (2,2.3), (3,3.3)} -- the middle 2 survives
```

## What the gate checks

`main.cpp` runs four scenarios -- a run of consecutive matches in the middle, a run right at the start, no matches at all, and every element matching -- and prints the resulting vector's size and contents after each. Your printed output is compared byte-for-byte against `ref.cpp`, compiled and run the same way: `exact_match == 1.0`. Any run of two or more consecutive matches exposes the skip-on-erase bug; the no-match and all-match scenarios catch off-by-one edge cases at the ends.
