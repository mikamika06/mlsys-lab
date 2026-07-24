#include "sol.hpp"

void filter_nodes(std::vector<DataNode>& nodes) {
    for (size_t i = 0; i < nodes.size(); ) {
        if (nodes[i].value < 0) {
            nodes.erase(nodes.begin() + (long)i);
            // do NOT advance -- the element that just shifted into slot i
            // must be examined too.
        } else {
            ++i;
        }
    }
}
