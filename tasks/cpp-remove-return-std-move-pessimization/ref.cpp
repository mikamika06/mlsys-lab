#include "sol.hpp"

Widget make_widget(int v) {
    Widget w(v);
    return w;  // plain local identifier -> NRVO-eligible, zero copies/moves
}
