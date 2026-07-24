## Context

`std::shared_ptr` manages an object with a reference count: the object is destroyed the instant the count reaches zero. If two objects hold `shared_ptr`s to *each other* -- a parent holding a child, and the child holding the parent back -- neither count can ever reach zero, even after every outside reference is dropped. That is a reference cycle, and it leaks for the lifetime of the process.

The standard fix is to make one direction of the cycle a `std::weak_ptr`. A `weak_ptr` observes an object without contributing to its strong reference count, so it can be `.lock()`ed while the object is alive but does not keep it alive by itself.

## Task

`Node` (declared in `sol.hpp`) builds a tree: `children` is a vector of strong (`shared_ptr`) references, and `parentSlot` is a `std::variant` that can hold either a `shared_ptr<Node>` or a `weak_ptr<Node>` back-edge to the parent.

Fix `Node::addChild` in `solve.cpp` so that the parent back-edge it stores in `child->parentSlot` is a `std::weak_ptr<Node>`, not a `std::shared_ptr<Node>`. (The shipped version stores a `shared_ptr`, creating exactly the cycle described above.)

`Node::parent()` is already correct: it returns whichever kind of reference `parentSlot` holds, locking it if it's a `weak_ptr`.

## Example

```cpp
auto root = std::make_shared<Node>("root");
auto child = std::make_shared<Node>("child");
root->addChild(child);          // child->parentSlot must become a weak_ptr<Node> to root

assert(child->parent() == root);   // still resolves while root is alive

root.reset();
child.reset();
// with a weak back-edge: both destructors ran, Node::dtorCount went up by 2.
// with a shared_ptr back-edge (the bug): neither ever runs -- a leak.
```

## What the gate checks

`main.cpp` builds a 3-node chain and a 13-node tree, drops every external `shared_ptr` into each, and prints `Node::dtorCount` (bumped by every real `~Node()` call) afterwards. With a weak back-edge the whole tree unwinds and `dtorCount` matches the node count; with a strong back-edge every parent/child pair keeps the other alive and `dtorCount` stays at 0. Your printed numbers are compared against `ref.cpp`, compiled and run the same way: `max_abs_err <= 1e-9`.
