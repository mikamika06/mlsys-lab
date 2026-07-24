#include "sol.hpp"

Node::Node(std::string n) : name(std::move(n)) {}
Node::~Node() { dtorCount++; }

void Node::addChild(const std::shared_ptr<Node>& child) {
    children.push_back(child);
    child->parentSlot = std::weak_ptr<Node>(shared_from_this());  // the fix: weak back-edge
}

std::shared_ptr<Node> Node::parent() const {
    if (std::holds_alternative<std::shared_ptr<Node>>(parentSlot))
        return std::get<std::shared_ptr<Node>>(parentSlot);
    if (std::holds_alternative<std::weak_ptr<Node>>(parentSlot))
        return std::get<std::weak_ptr<Node>>(parentSlot).lock();
    return nullptr;
}
