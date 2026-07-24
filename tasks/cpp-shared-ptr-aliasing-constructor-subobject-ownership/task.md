## Context

`std::shared_ptr<T>` has a little-known *aliasing constructor*:

```cpp
template <class Y>
shared_ptr(const shared_ptr<Y>& owner, T* ptr) noexcept;
```

The resulting `shared_ptr<T>` *shares ownership* with `owner` — it uses the same
control block and bumps the same reference count — but its stored pointer
(`get()`) is whatever `ptr` you pass, which need not be `owner.get()`. The
control block still deletes the *original* object (the `owner`'s), never `ptr`.

This is exactly how you hand out a pointer to a **subobject** while keeping its
**parent** alive. Given a `shared_ptr<Parent>`, you can return a
`shared_ptr<Child>` aimed at `&parent->child`. The parent's reference count now
counts that alias too, so:

- dropping every `shared_ptr<Parent>` does **not** destroy the parent, and
- the parent's destructor runs only when the **last alias** dies.

A naive `std::shared_ptr<Child>(&parent->child)` would instead create a *new*
control block that tries to `delete` a subobject — undefined behavior — and
would not keep the parent alive at all.

## Task

Implement, in `solve.cpp`:

```cpp
std::shared_ptr<Child> alias_child(const std::shared_ptr<Parent>& parent);
```

It must return a `shared_ptr<Child>` that points to `parent->child` while
sharing ownership with `parent`, using the aliasing constructor. The returned
pointer (and any copy of it) must keep the whole `Parent` alive; when the last
such alias is destroyed, the `Parent` destructor runs.

`Parent`, `Child`, and the destruction log `g_dtor_log` are declared in
`sol.hpp`; do not modify them.

## Example

```cpp
auto p = std::make_shared<Parent>();      // id=7, child.value=42
auto a = alias_child(p);                  // a.use_count() == 2
p.reset();                                // Parent NOT destroyed yet
assert(a.use_count() == 1);
assert(a->value == 42);                   // subobject still alive
a.reset();                                // NOW the Parent destructor fires
```

## What the gate checks

The fixed driver in `main.cpp` builds a `Parent` (id 7, child value 42), takes
an alias, and prints seven integers: the shared use-count while both owners are
alive (2), the value read through the alias (42), the number of destructors
fired right after the last `shared_ptr<Parent>` is reset (0 — the parent is kept
alive by the alias), the alias use-count then (1), the value still readable
through the alias (42), the destructor count after the alias is reset (1), and
the id of the destroyed parent (7).

The gate is `exact_match`: your program's printed lines must match the reference
solution's output exactly. A pointer that does not co-own the parent (or is
null) changes the destruction-order log and fails.
