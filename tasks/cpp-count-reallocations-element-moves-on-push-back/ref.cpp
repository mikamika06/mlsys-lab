#include "sol.hpp"
#include <new>
#include <utility>

long simulate_vector_pushes(int N, int initial_capacity, int growth_factor) {
    long capacity = initial_capacity;
    long size = 0;
    long reallocs = 0;

    Element* buf = capacity > 0
        ? static_cast<Element*>(::operator new(sizeof(Element) * (size_t)capacity))
        : nullptr;

    for (int i = 0; i < N; ++i) {
        if (size == capacity) {
            long new_capacity = (capacity == 0) ? 1 : capacity * growth_factor;
            Element* new_buf = static_cast<Element*>(
                ::operator new(sizeof(Element) * (size_t)new_capacity));
            for (long j = 0; j < size; ++j) {
                ::new (static_cast<void*>(new_buf + j)) Element(std::move(buf[j]));
                buf[j].~Element();
            }
            if (buf) ::operator delete(buf);
            buf = new_buf;
            capacity = new_capacity;
            ++reallocs;
        }

        Element tmp;
        ::new (static_cast<void*>(buf + size)) Element(std::move(tmp));
        ++size;
    }

    for (long j = 0; j < size; ++j) buf[j].~Element();
    if (buf) ::operator delete(buf);

    return reallocs;
}
