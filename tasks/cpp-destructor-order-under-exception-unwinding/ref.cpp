#include "sol.hpp"
#include <utility>
#include <vector>

long run_trace(const Stmt* stmts, int n, int* out_ids, int* out_count) {
    // Stack of open scopes; each scope holds its live (id, bytes) objects
    // in construction order.
    std::vector<std::vector<std::pair<int, int>>> scopes;
    scopes.push_back({});

    for (int i = 0; i < n; i++) {
        const Stmt& s = stmts[i];
        switch (s.kind) {
            case BEGIN:
                scopes.push_back({});
                break;
            case END:
                scopes.pop_back();  // normal destruction: not part of an unwind
                break;
            case CONSTRUCT:
                scopes.back().push_back({s.id, s.bytes});
                break;
            case THROW: {
                int cnt = 0;
                long total = 0;
                // Innermost scope first, and within each scope, most
                // recently constructed first -- overall reverse
                // chronological order across the whole live set, since
                // scopes were opened in chronological order too.
                for (int sc = (int)scopes.size() - 1; sc >= 0; sc--) {
                    auto& live = scopes[sc];
                    for (int k = (int)live.size() - 1; k >= 0; k--) {
                        out_ids[cnt++] = live[k].first;
                        total += live[k].second;
                    }
                }
                *out_count = cnt;
                return total;
            }
        }
    }
    *out_count = 0;
    return 0;
}
