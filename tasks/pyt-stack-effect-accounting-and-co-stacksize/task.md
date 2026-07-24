## Context

CPython bytecode instructions manipulate a value stack. Each opcode has a stack
effect that describes how many values it consumes and produces. For an
instruction sequence $I_1, I_2, \dots, I_n$, the running stack depth can be
computed as

$$
s_k = s_{k-1} + \Delta(I_k),
$$

where $\Delta(I_k)$ is the net stack effect of instruction $I_k$.

The maximum value reached by $s_k$ is the stack depth required to execute the
sequence. CPython stores this requirement in a code object's `co_stacksize`
field. The stack accounting problem is therefore to derive the final net effect

$$
\Delta_{\mathrm{total}} = \sum_{k=1}^{n} \Delta(I_k)
$$

and the maximum intermediate depth

$$
s_{\mathrm{max}} = \max_k s_k .
$$

The `dis` module exposes CPython's own opcode stack-effect table, including
conditional effects for instructions whose behavior depends on jump branches.

## Task

Implement `stack_account(source)`:

```python
def stack_account(source: str) -> tuple[int, int]:
    ...
```

The function receives a Python source string containing a single module-level
code object. Compile the source, inspect its bytecode instructions, and return:

```python
(net_stack_effect, max_stack_depth)
```

Use CPython's `dis` module to obtain instructions and stack effects. The
calculated net effect should include all instructions in the compiled top-level
code object. For conditional stack effects, use the larger possible stack
change when computing the maximum depth.

The return values must be Python integers.

## Example

```python
source = "x = 1 + 2"

result = stack_account(source)
# result is equivalent to:
# (
#   sum(dis.stack_effect(i.opcode, i.arg) for i in dis.get_instructions(compile(source, '<x>', 'exec'))),
#   max running stack depth from those effects
# )
```

## What the gate checks

The gate builds several real Python code objects and uses CPython's `dis`
module as the oracle for each instruction's stack effect. The returned pair
must exactly match the oracle-computed final stack change and maximum simulated
stack depth.
