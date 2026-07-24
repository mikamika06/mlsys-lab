## Context

Byte‑Pair Encoding (BPE) is a subword tokenisation technique that starts from an initial vocabulary of single characters and repeatedly merges the most frequent adjacent pair of symbols into a new symbol. After a sequence of merge operations, each input string can be represented as a list of tokens drawn from the final vocabulary.

Let $S$ be a corpus consisting of strings $s_1,\dots,s_n$.  
Let $\mathcal{M} = \bigl[(a_1,b_1),\dots,(a_m,b_m)\bigr]$ be an ordered list of merge pairs.  
The tokenisation process is:

1. Initialise the token list for each string as its individual characters (including whitespace).
2. For each merge pair $(a,b)$ in $\mathcal{M}$, scan every token list and replace any adjacent occurrence of $[\,a,\;b\,]$ by the single token $ab$.
3. After all merges have been applied, the total number of tokens is the sum over all strings of the length of their final token lists.

The task below asks you to implement a function that returns this total count for an arbitrary corpus and merge list.

## Task

Implement the following function:

```python
def total_token_count(corpus: list[str], merges: list[tuple[str, str]]) -> int:
    """
    Return the total number of BPE tokens produced when applying the given
    ordered list of merge pairs to each string in `corpus`.

    Parameters
    ----------
    corpus : list[str]
        A list of input strings.  Each string may contain any Unicode
        characters, including whitespace.
    merges : list[tuple[str, str]]
        An ordered list of merge operations.  Each pair `(a, b)` indicates
        that the adjacent tokens `a` followed by `b` should be replaced by
        the single token `ab`.  The order matters: earlier merges are applied
        before later ones.

    Returns
    -------
    int
        The total number of tokens across all strings after all merges have
        been performed.
    """
```

The implementation must use only Python built‑ins and standard library modules.  No external packages are allowed.

## Example

```python
corpus = ["ab", "ba"]
merges = [("a", "b")]

# Initially: ['a', 'b'] for each string → total tokens = 4
# After applying ("a","b"): ['ab'] for each string → total tokens = 2

print(total_token_count(corpus, merges))  # Output: 2
```

```python
corpus = ["abc"]
merges = [("a", "b"), ("ab", "c")]

# Step 1: ['a', 'b', 'c']
# Step 2: merge ('a','b') → ['ab', 'c']
# Step 3: merge ('ab','c') → ['abc']

print(total_token_count(corpus, merges))  # Output: 1
```

## What the gate checks

The grader computes a reference token count by applying the same BPE algorithm to each test case.  
Your implementation must return an integer that matches this reference exactly; otherwise the `exact_match` gate fails.
