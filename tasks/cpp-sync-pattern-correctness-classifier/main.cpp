#include "sol.hpp"
#include <cstdio>

int main() {
    std::vector<SyncPattern> patterns(10);

    patterns[0].reads = {"count"};
    patterns[0].writes = {"count"};
    patterns[0].atomics = {"count"};

    patterns[1].reads = {"value"};
    patterns[1].writes = {"value"};

    patterns[2].locks = {{"A", "B"}, {"B", "A"}};
    patterns[2].lock_edges = {{"A", "B"}, {"B", "A"}};

    patterns[3].reads = {"node"};
    patterns[3].locks = {{"tree", "tree"}};

    patterns[4].reads = {"flag"};
    patterns[4].writes = {"data"};
    patterns[4].atomics = {"flag"};

    patterns[5].reads = {"queue"};
    patterns[5].writes = {"queue"};
    patterns[5].locks = {{"queue_lock", "queue_lock"}};

    patterns[6].reads = {"ready"};
    patterns[6].writes = {"ready"};
    patterns[6].atomics = {"ready"};

    patterns[7].reads = {"x"};
    patterns[7].writes = {"y"};

    patterns[8].writes = {"counter"};

    patterns[9].reads = {"a", "b"};
    patterns[9].lock_edges = {{"a", "b"}, {"b", "c"}, {"c", "a"}};

    std::vector<std::string> out(10);
    classify_sync_patterns(patterns.data(), 10, out.data());
    for (int i = 0; i < 10; i++) {
        printf("%s\n", out[i].c_str());
    }
    return 0;
}
