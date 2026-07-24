#include "sol.hpp"

int Widget::read() {
    calls++;
    return calls;
}

// BUG: `calls` is `mutable`, so the compiler allows this write inside a
// const member function -- but a well-behaved read() const must still leave
// observable state unchanged. This version silently mutates `calls` on
// every "read", making the const overload behave exactly like the
// non-const one and defeating the whole point of offering a const view.
int Widget::read() const {
    calls++;
    return calls;
}
