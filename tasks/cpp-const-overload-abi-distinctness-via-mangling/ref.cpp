#include "sol.hpp"

int Widget::read() {
    calls++;
    return calls;
}

int Widget::read() const {
    return calls;   // read-only, even though `calls` is `mutable` and could be written
}
