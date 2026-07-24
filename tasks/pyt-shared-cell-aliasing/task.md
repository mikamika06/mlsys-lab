## Context

A Python closure can capture a variable from an enclosing function scope. The captured variable is stored in a cell object, and multiple inner functions can reference the same cell.

Consider an enclosing scope with a variable $x$. If two functions both close over $x$, then a mutation performed through one function is visible when the other function reads the variable. The important property is that both functions reference the same storage location rather than independent copies.

The observable behavior can be described as a sequence of state transitions. If the initial state is $x_0$ and an update changes the state by $\Delta$, then a reader of the same cell observes

$$x_1 = x_0 + \Delta.$$

This task tests whether a closure is built with shared cell state.

## Task

Implement `shared_cell_trace()`:

```python
def shared_cell_trace():
    ...
```

The function must create two inner functions that close over the same nonlocal variable. One inner function must mutate that variable and the other must read it.

Return a tuple containing the values observed by the reader after the mutations. The returned tuple must exactly match the behavior of a closure where both functions share one enclosing cell.

Do not use global variables. The state must live in the enclosing function scope.

## Example

```python
result = shared_cell_trace()

# result contains the sequence of values observed from the shared closure state.
```

## What the gate checks

The gate builds an independent reference implementation using real Python closure behavior. It creates two inner functions sharing one enclosing variable, mutates the variable through one function, and records the values returned by the other function.

Your implementation is called and its return value is compared with the oracle result using exact equality. Implementations that create separate variables, return only constants, or avoid shared closure state will fail.
