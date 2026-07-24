// FIXED driver. Builds the 20 declarations from task.md for real (5 scalar
// types x {automatic, static, thread_local, dynamic}) and derives the
// ground truth for each PURELY from observed runtime behaviour -- never a
// hardcoded label table:
//
//  - automatic: two overlapping activations of the same local variable
//    (forced via real recursion, so both are alive at once) must land at
//    DIFFERENT addresses -- that's what "a fresh object per activation"
//    means.
//  - static / thread_local: call the same function from two REAL OS
//    threads (std::thread) and compare the addresses it returns. A plain
//    `static` local is one shared object -> same address in both threads.
//    A `thread_local` local is a per-thread instance -> different address
//    in each thread. The language guarantees both, so this is a real,
//    deterministic test, not a heuristic.
//  - dynamic: a global operator new/delete override records every
//    heap-allocated address; a `new T` pointer is "dynamic" iff its
//    address was actually handed out by the allocator.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#include "sol.hpp"

// ---- heap-allocation tracking, to recognize `new T` ----
static void* g_heap_ptrs[64];
static int g_heap_count = 0;

void* operator new(std::size_t sz) {
    void* p = std::malloc(sz);
    if (g_heap_count < 64) g_heap_ptrs[g_heap_count++] = p;
    return p;
}
void operator delete(void* p) noexcept { std::free(p); }

static bool is_heap_ptr(const void* p) {
    for (int i = 0; i < g_heap_count; i++) {
        if (g_heap_ptrs[i] == p) return true;
    }
    return false;
}

// ---- automatic: forced-overlapping recursive activations ----
template <typename T>
struct AutoProbe {
    static void go(int depth, uintptr_t* out) {
        T x{};
        out[depth] = (uintptr_t)&x;
        if (depth == 0) go(1, out);
    }
};

template <typename T>
static bool probe_is_automatic() {
    uintptr_t out[2] = {0, 0};
    AutoProbe<T>::go(0, out);
    return out[0] != out[1];
}

// ---- static / thread_local: cross-thread address comparison ----
template <typename T>
static uintptr_t static_local_addr() {
    static T x{};
    return (uintptr_t)&x;
}

template <typename T>
static uintptr_t thread_local_addr() {
    thread_local T x{};
    return (uintptr_t)&x;
}

template <typename T>
static bool probe_is_plain_static() {
    uintptr_t main_addr = static_local_addr<T>();
    uintptr_t other_addr = 0;
    std::thread th([&]() { other_addr = static_local_addr<T>(); });
    th.join();
    return main_addr == other_addr;  // one shared instance across threads
}

template <typename T>
static bool probe_is_thread_local() {
    uintptr_t main_addr = thread_local_addr<T>();
    uintptr_t other_addr = 0;
    std::thread th([&]() { other_addr = thread_local_addr<T>(); });
    th.join();
    return main_addr != other_addr;  // distinct instance per thread
}

// ---- dynamic: was this pointer actually handed out by `new`? ----
template <typename T>
static bool probe_is_dynamic() {
    T* p = new T;
    bool result = is_heap_ptr(p);
    delete p;
    return result;
}

template <typename T>
static void classify_group(std::string out[4]) {
    out[0] = probe_is_automatic<T>() ? "automatic" : "UNKNOWN";
    out[1] = probe_is_plain_static<T>() ? "static" : "UNKNOWN";
    out[2] = probe_is_thread_local<T>() ? "thread" : "UNKNOWN";
    out[3] = probe_is_dynamic<T>() ? "dynamic" : "UNKNOWN";
}

int main() {
    std::string truth[20];
    classify_group<int>(&truth[0]);
    classify_group<double>(&truth[4]);
    classify_group<char>(&truth[8]);
    classify_group<short>(&truth[12]);
    classify_group<long long>(&truth[16]);

    std::string pred[20];
    name_storage_durations(pred);

    int matches = 0;
    for (int i = 0; i < 20; i++) {
        int ok = (pred[i] == truth[i]) ? 1 : 0;
        matches += ok;
        printf("%d %d\n", i + 1, ok);
    }
    printf("matches %d\n", matches);
    return 0;
}
