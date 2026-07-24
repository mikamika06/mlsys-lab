#include "sol.hpp"

// BUG: unconditionally advances `i` even after an erase, so the element
// that just shifted into slot i (from erasing slot i) never gets
// examined -- it silently survives whenever two or more negative values
// are adjacent.
void filter_nodes(std::vector<DataNode>& nodes) {
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].value < 0) {
            nodes.erase(nodes.begin() + (long)i);
        }
    }
}
