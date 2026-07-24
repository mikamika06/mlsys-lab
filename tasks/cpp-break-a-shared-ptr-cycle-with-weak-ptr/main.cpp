#include <cstdio>
#include <string>
#include "sol.hpp"

int Node::dtorCount = 0;

int main() {
    // 1. a 3-node chain: root -> child1 -> child2, then every external
    //    shared_ptr (including the two returned by parent()) is dropped.
    Node::dtorCount = 0;
    {
        auto root = std::make_shared<Node>("root");
        auto child1 = std::make_shared<Node>("child1");
        auto child2 = std::make_shared<Node>("child2");
        root->addChild(child1);
        child1->addChild(child2);

        auto p1 = child1->parent();
        auto p2 = child2->parent();
        printf("links_ok %d\n", (p1 == root && p2 == child1) ? 1 : 0);
    }
    printf("after_3node %d\n", Node::dtorCount);

    // 2. a wider tree: root with 4 children, each with 2 grandchildren
    //    (13 nodes total), everything built then let go out of scope.
    Node::dtorCount = 0;
    {
        auto root = std::make_shared<Node>("root");
        for (int i = 0; i < 4; i++) {
            auto c = std::make_shared<Node>("c" + std::to_string(i));
            root->addChild(c);
            for (int j = 0; j < 2; j++) {
                auto g = std::make_shared<Node>("g");
                c->addChild(g);
            }
        }
    }
    printf("after_wide %d\n", Node::dtorCount);

    // 3. a node with no parent must report parent() == nullptr, and must
    //    still be destroyed normally (no cycle involved at all here).
    Node::dtorCount = 0;
    {
        auto lonely = std::make_shared<Node>("lonely");
        printf("no_parent %d\n", (lonely->parent() == nullptr) ? 1 : 0);
    }
    printf("after_lonely %d\n", Node::dtorCount);

    return 0;
}
