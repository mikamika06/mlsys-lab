#include <cstdio>
#include <utility>
#include <vector>
#include "sol.hpp"

// ---------------------------------------------------------------- instrumented heap
namespace heap {
    static std::vector<int> values;   // index+1 == id; -1 marks a released slot
    static int dfree = 0;

    int alloc(int value) { values.push_back(value); return (int)values.size(); }

    void release(int id) {
        if (id == 0) return;                                   // delete nullptr
        if (id < 0 || id > (int)values.size() || values[id - 1] == -1) { dfree++; return; }
        values[id - 1] = -1;
    }

    int read(int id) {
        if (id <= 0 || id > (int)values.size() || values[id - 1] == -1) return 0;
        return values[id - 1];
    }

    int live() { int n = 0; for (int v : values) if (v != -1) n++; return n; }
    int doubleFrees() { return dfree; }
    void reset() { values.clear(); dfree = 0; }
}

static void line(const char* tag, int a, int b, int c) {
    printf("%s %d %d %d\n", tag, a, b, c);
}

int main() {
    // 1. move construction: the source must end up owning nothing
    heap::reset();
    {
        Buffer a(42);
        Buffer b(std::move(a));
        line("move_ctor", b.value(), a.id() == 0 ? 1 : 0, heap::live());
    }
    line("after_scope", heap::live(), heap::doubleFrees(), 0);

    // 2. copy construction is a deep copy: two live ids, independent values
    heap::reset();
    {
        Buffer a(7);
        Buffer b(a);
        line("copy_ctor", b.value(), a.id() != b.id() ? 1 : 0, heap::live());
    }
    line("after_scope", heap::live(), heap::doubleFrees(), 0);

    // 3. move assignment releases what the target already owned
    heap::reset();
    {
        Buffer a(1), b(2);
        b = std::move(a);
        line("move_assign", b.value(), a.id() == 0 ? 1 : 0, heap::live());
    }
    line("after_scope", heap::live(), heap::doubleFrees(), 0);

    // 4. copy assignment is deep and self-assignment is harmless
    heap::reset();
    {
        Buffer a(5), b(9);
        b = a;
        int deep = (a.id() != b.id() && b.value() == 5) ? 1 : 0;
        a = a;                                       // must not destroy a
        line("copy_assign", deep, a.value(), heap::live());
    }
    line("after_scope", heap::live(), heap::doubleFrees(), 0);

    // 5. a vector reallocation moves every element; nothing may be freed twice
    heap::reset();
    {
        std::vector<Buffer> v;
        for (int i = 0; i < 8; i++) v.push_back(Buffer(100 + i));
        int sum = 0;
        for (const auto& b : v) sum += b.value();
        line("vector", sum, (int)v.size(), heap::doubleFrees());
    }
    line("after_scope", heap::live(), heap::doubleFrees(), 0);

    return 0;
}
