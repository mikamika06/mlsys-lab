#pragma once
#include <memory>
#include <string>
#include <variant>
#include <vector>

// A parent/child tree of std::shared_ptr<Node>. `children` holds a STRONG
// (shared_ptr) reference to each child -- that part is correct and fixed.
// `parentSlot` is where the back-edge (child -> parent) is recorded; it can
// hold either a shared_ptr<Node> or a weak_ptr<Node> (that's the whole
// point -- it's addChild's job to decide which). If the back-edge is a
// shared_ptr, the parent and child each keep the other's refcount above
// zero forever: neither is ever destroyed, even after every external
// shared_ptr into the tree is dropped. The fix is to store a weak_ptr.
//
// dtorCount is bumped by every ~Node() call. The grader builds a tree, drops
// every external shared_ptr to it, and compares dtorCount to the number of
// nodes it built -- a strong back-edge leaves dtorCount at 0 (a leak).
struct Node : std::enable_shared_from_this<Node> {
    static int dtorCount;

    std::string name;
    std::vector<std::shared_ptr<Node>> children;
    std::variant<std::monostate, std::shared_ptr<Node>, std::weak_ptr<Node>> parentSlot;

    explicit Node(std::string n);
    ~Node();

    // Records `child` as a child of `this` (strong ref, in `children`) and
    // records `this` as `child`'s parent (must be a WEAK ref, in
    // `child->parentSlot`).
    void addChild(const std::shared_ptr<Node>& child);

    // Returns the parent as a shared_ptr (locking parentSlot if it holds a
    // weak_ptr), or nullptr if there is no parent or it has been freed.
    std::shared_ptr<Node> parent() const;
};
