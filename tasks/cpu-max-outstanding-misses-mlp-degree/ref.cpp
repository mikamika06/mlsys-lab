#include "sol.hpp"
#include <vector>
#include <queue>
#include <algorithm>

int mlp_degree(int n, const int* dep_from, const int* dep_to, int num_edges) {
    std::vector<std::vector<int>> adj(n);
    std::vector<int> indeg(n, 0);
    for (int i = 0; i < num_edges; ++i) {
        adj[dep_from[i]].push_back(dep_to[i]);
        ++indeg[dep_to[i]];
    }

    std::vector<int> depth(n, 0);
    std::queue<int> q;
    for (int v = 0; v < n; ++v) {
        if (indeg[v] == 0) q.push(v);
    }
    while (!q.empty()) {
        int v = q.front();
        q.pop();
        for (int w : adj[v]) {
            depth[w] = std::max(depth[w], depth[v] + 1);
            if (--indeg[w] == 0) q.push(w);
        }
    }

    std::vector<int> count_at_depth(n, 0);
    for (int v = 0; v < n; ++v) {
        count_at_depth[depth[v]] += 1;
    }

    int mx = 0;
    for (int d = 0; d < n; ++d) {
        mx = std::max(mx, count_at_depth[d]);
    }
    return mx;
}
