#include <cstdio>
#include <vector>
#include "sol.hpp"

int g_copy_count = 0;
int g_move_count = 0;

Buffer::Buffer() : id(0), data(nullptr), capacity(0), size(0) {}

Buffer::Buffer(const Buffer& other)
    : id(other.id), data(nullptr), capacity(other.capacity), size(other.size) {
    g_copy_count++;
    if (other.data != nullptr && other.size > 0) {
        data = new double[(size_t)other.size];
        for (long i = 0; i < other.size; i++) data[i] = other.data[i];
    }
}

Buffer::Buffer(Buffer&& other) noexcept
    : id(other.id), data(other.data), capacity(other.capacity), size(other.size) {
    g_move_count++;
    other.data = nullptr;
    other.capacity = 0;
    other.size = 0;
}

Buffer& Buffer::operator=(const Buffer& other) {
    g_copy_count++;
    if (this != &other) {
        delete[] data;
        data = nullptr;
        id = other.id;
        capacity = other.capacity;
        size = other.size;
        if (other.data != nullptr && other.size > 0) {
            data = new double[(size_t)other.size];
            for (long i = 0; i < other.size; i++) data[i] = other.data[i];
        }
    }
    return *this;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    g_move_count++;
    if (this != &other) {
        delete[] data;
        id = other.id;
        data = other.data;
        capacity = other.capacity;
        size = other.size;
        other.data = nullptr;
        other.capacity = 0;
        other.size = 0;
    }
    return *this;
}

Buffer::~Buffer() {
    delete[] data;
}

static void run_case(const Op* ops, int n) {
    g_copy_count = 0;
    g_move_count = 0;
    std::vector<Buffer> vec;
    run_pipeline(ops, n, vec);
    printf("copies=%d moves=%d final_size=%d\n", g_copy_count, g_move_count, (int)vec.size());
}

// FIXED driver. Two deterministic op sequences that exercise every op kind
// plus the reallocation moves std::vector performs on its own as it grows.
int main() {
    const Op case1[] = {
        {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},  // 5x push_temp
        {1, 0, 0}, {1, 0, 0}, {1, 0, 0},                        // 3x push_lvalue
        {2, 0, 1},                                              // copy_assign dst=0 src=1
        {3, 2, 3},                                              // move_assign dst=2 src=3
    };
    run_case(case1, (int)(sizeof(case1) / sizeof(case1[0])));

    const Op case2[] = {
        {1, 0, 0}, {1, 0, 0},                                   // 2x push_lvalue
        {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},              // 4x push_temp
        {3, 0, 5},                                              // move_assign dst=0 src=5
        {2, 4, 1},                                              // copy_assign dst=4 src=1
        {3, 2, 0},                                              // move_assign dst=2 src=0
    };
    run_case(case2, (int)(sizeof(case2) / sizeof(case2[0])));

    return 0;
}
