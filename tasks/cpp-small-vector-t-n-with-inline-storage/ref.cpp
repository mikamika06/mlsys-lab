#include "sol.hpp"
#include <new>
#include <utility>

// Correct reference implementation: inline storage until CAP, heap spill after.

SmallVector::SmallVector()
    : data(reinterpret_cast<Tracked*>(inbuf)), sz(0), cap(CAP) {}

SmallVector::~SmallVector() {
    for (int i = 0; i < sz; ++i)
        data[i].~Tracked();                       // manual destruction
    if (data != reinterpret_cast<Tracked*>(inbuf))
        ::operator delete(data);                  // free heap spill
}

void SmallVector::push_back(long v) {
    if (sz == cap) {
        int newcap = cap * 2;
        Tracked* nd = static_cast<Tracked*>(::operator new(sizeof(Tracked) * newcap));
        for (int i = 0; i < sz; ++i) {
            ::new (static_cast<void*>(nd + i)) Tracked(data[i]);  // copy-construct into new
            data[i].~Tracked();                                   // destroy the old one
        }
        if (data != reinterpret_cast<Tracked*>(inbuf))
            ::operator delete(data);              // free the old heap block, if any
        data = nd;
        cap = newcap;
    }
    ::new (static_cast<void*>(data + sz)) Tracked(v);  // placement-new the new element
    ++sz;
}

long SmallVector::sum() const {
    long s = 0;
    for (int i = 0; i < sz; ++i)
        s += data[i].value;
    return s;
}

int SmallVector::size() const { return sz; }

bool SmallVector::spilled() const {
    return data != reinterpret_cast<const Tracked*>(inbuf);
}
