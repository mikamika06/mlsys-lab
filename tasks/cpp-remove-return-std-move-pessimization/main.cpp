#include <cstdio>
#include "sol.hpp"

long g_copy_count = 0;
long g_move_count = 0;

Widget::Widget() : value(0) {}
Widget::Widget(int v) : value(v) {}
Widget::Widget(const Widget& o) : value(o.value) { ++g_copy_count; }
Widget::Widget(Widget&& o) noexcept : value(o.value) { ++g_move_count; }

// FIXED driver. Do not edit. Calls make_widget for two fixed values,
// resetting the counters between calls, and prints the resulting value
// plus the real observed copy/move counts.
int main() {
    int values[] = {42, -7};
    for (int v : values) {
        g_copy_count = 0;
        g_move_count = 0;
        Widget w = make_widget(v);
        printf("value=%d copy_count=%ld move_count=%ld\n",
               w.value, g_copy_count, g_move_count);
    }
    return 0;
}
