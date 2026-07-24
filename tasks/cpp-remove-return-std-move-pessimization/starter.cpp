#include "sol.hpp"
#include <utility>

// BUG: std::move(w) casts the local to an rvalue reference, so the return
// expression is no longer a plain local variable identifier -- this
// blocks NRVO and forces a move construction (move_count == 1) instead
// of the zero-copy, zero-move construction NRVO would give.
Widget make_widget(int v) {
    Widget w(v);
    return std::move(w);
}
