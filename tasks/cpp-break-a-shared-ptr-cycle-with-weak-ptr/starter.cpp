#include "sol.hpp"

Node::Node(std::string n) : name(std::move(n)) {}
Node::~Node() { dtorCount++; }

// BUG: the back-edge is stored as a full shared_ptr, so parent and child
// each keep the other alive forever -- a reference cycle that never frees.
void Node::addChild(const std::shared_ptr<Node>& child) {
    children.push_back(child);
    child->parentSlot = shared_from_this();   // should be a weak_ptr
}

std::shared_ptr<Node> Node::parent() const {
    if (std::holds_alternative<std::shared_ptr<Node>>(parentSlot))
        return std::get<std::shared_ptr<Node>>(parentSlot);
    if (std::holds_alternative<std::weak_ptr<Node>>(parentSlot))
        return std::get<std::weak_ptr<Node>>(parentSlot).lock();
    return nullptr;
}
