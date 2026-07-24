## Context

The GPT‑2 language model uses a byte‑pair encoding (BPE) tokenizer.  
Before the BPE step, the raw text is split into *pre‑tokens* by a regular expression that groups together contiguous sequences of either non‑ASCII characters, word characters, or any other single non‑whitespace character.  The official pattern used in the HuggingFace `tokenizers` library is

$$
\texttt{(?:(?:[^\x00-\x7F]+)|(?:\\w+)|(?:[^\\s\\w]))}
$$

Applying this pattern with `re.findall` yields a list of strings that will later be fed to the BPE merge table.

## Task

Implement the function

```python
def split_gpt2_pre_tokenizer(text: str) -> List[str]:
    ...
```

It should return the list of pre‑token splits produced by the regex above.  The output must be a Python `list` of `str`, preserving the order in which the matches appear in the input.

## Example

```python
>>> split_gpt2_pre_tokenizer("Hello, world! Café 😊")
['Hello', ',', ' ', 'world', '!', ' ', 'Café', ' ', '😊']
```

The comma and exclamation mark are captured as separate tokens; spaces are also returned because the regex matches any non‑whitespace character individually.

## What the gate checks

* **Exact match** – The list of strings produced by your implementation must be identical to that produced by a reference implementation based on the official regex.  Any deviation, including missing or reordered tokens, causes the gate to fail.

No additional performance constraints are imposed; correctness is the sole criterion.
