#include "sol.hpp"
#include <map>
#include <set>

namespace {

bool visit(const std::string& node, std::set<std::string>& stack,
           const std::map<std::string, std::vector<std::string>>& graph) {
    if (stack.count(node)) return true;
    stack.insert(node);
    auto it = graph.find(node);
    if (it != graph.end()) {
        for (const auto& nxt : it->second) {
            if (visit(nxt, stack, graph)) return true;
        }
    }
    stack.erase(node);
    return false;
}

std::string classify_one(const SyncPattern& p) {
    for (const auto& lock : p.locks) {
        if (lock.first == lock.second) return "deadlock";
    }

    std::map<std::string, std::vector<std::string>> graph;
    for (const auto& edge : p.lock_edges) {
        graph[edge.first].push_back(edge.second);
    }
    for (const auto& kv : graph) {
        std::set<std::string> stack;
        if (visit(kv.first, stack, graph)) return "deadlock";
    }

    std::set<std::string> writes(p.writes.begin(), p.writes.end());
    std::set<std::string> reads(p.reads.begin(), p.reads.end());
    std::set<std::string> atomics(p.atomics.begin(), p.atomics.end());

    std::set<std::string> shared;
    for (const auto& s : writes) shared.insert(s);
    for (const auto& s : reads) shared.insert(s);
    for (const auto& s : atomics) shared.erase(s);

    for (const auto& item : writes) {
        if (shared.count(item) && reads.count(item)) return "data-race";
    }

    return "ok";
}

} // namespace

void classify_sync_patterns(const SyncPattern* patterns, int n, std::string* out) {
    for (int i = 0; i < n; i++) {
        out[i] = classify_one(patterns[i]);
    }
}
