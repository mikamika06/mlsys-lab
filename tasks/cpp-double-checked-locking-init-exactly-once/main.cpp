// Fixed driver: scripts the classic double-checked-locking worst case
// deterministically (no real OS threads, no timing) — three callers all
// observe "not ready" on the fast path before any of them reaches the
// lock, then all three fall through to the slow path in sequence (as the
// mutex would force in a real race).
#include "sol.hpp"
#include <cstdio>

int main() {
    SingletonState s;

    bool fc1 = fast_check(s);
    bool fc2 = fast_check(s);
    bool fc3 = fast_check(s);

    bool did1 = try_init(s);
    bool did2 = try_init(s);
    bool did3 = try_init(s);

    printf("%d %d %d\n", fc1 ? 1 : 0, fc2 ? 1 : 0, fc3 ? 1 : 0);
    printf("%d %d %d\n", did1 ? 1 : 0, did2 ? 1 : 0, did3 ? 1 : 0);
    printf("%d\n", s.init_count);
    printf("%d\n", s.ready.load() ? 1 : 0);
    return 0;
}
