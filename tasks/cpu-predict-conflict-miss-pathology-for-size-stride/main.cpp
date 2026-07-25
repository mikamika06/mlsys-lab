#include <cstdio>
#include <list>
#include <unordered_map>
#include <vector>
#include "sol.hpp"

// ============================================================================
// FIXED driver: a real NUM_SETS-way-indexed, WAYS-associative LRU cache
// (independent ground truth -- does not read your prediction). Each of the
// NUM_SETS sets is its own WAYS-line fully-associative LRU pool.
// ============================================================================
namespace {

class LruSet {
public:
    explicit LruSet(int cap) : cap_(cap) {}
    bool contains(long line) const { return map_.find(line) != map_.end(); }
    void touch(long line) {
        auto it = map_.find(line);
        order_.erase(it->second);
        order_.push_front(line);
        it->second = order_.begin();
    }
    // Returns true if this was a MISS (line inserted fresh).
    bool access(long line) {
        auto it = map_.find(line);
        if (it != map_.end()) {
            order_.erase(it->second);
            order_.push_front(line);
            it->second = order_.begin();
            return false;
        }
        if (static_cast<int>(order_.size()) >= cap_) {
            long evict = order_.back();
            order_.pop_back();
            map_.erase(evict);
        }
        order_.push_front(line);
        map_[line] = order_.begin();
        return true;
    }

private:
    int cap_;
    std::list<long> order_;
    std::unordered_map<long, std::list<long>::iterator> map_;
};

// Run one sweep of addresses 0, stride, 2*stride, ... (n = array_size /
// stride elements) through a fresh NUM_SETS x WAYS set-associative cache
// TWICE (same order both times) and report whether the second sweep added
// any misses beyond the first (empirical "pathological" ground truth).
int observed_pathological(long array_size, long stride) {
    std::vector<LruSet> sets;
    sets.reserve(NUM_SETS);
    for (int i = 0; i < NUM_SETS; ++i) sets.emplace_back(WAYS);

    long n = array_size / stride;
    long misses1 = 0, misses2 = 0;
    for (long k = 0; k < n; ++k) {
        long addr = k * stride;
        long line = addr / LINE_BYTES;
        int s = static_cast<int>(line % NUM_SETS);
        if (sets[s].access(line)) ++misses1;
    }
    for (long k = 0; k < n; ++k) {
        long addr = k * stride;
        long line = addr / LINE_BYTES;
        int s = static_cast<int>(line % NUM_SETS);
        if (sets[s].access(line)) ++misses2;
    }
    return (misses2 > 0) ? 1 : 0;
}

struct Case {
    long array_size;
    long stride;
};

constexpr Case CASES[] = {
    {32768, 64},
    {16384, 2048},
    {12288, 192},
    {65536, 4096},
    {65536, 128},
    {65536, 1024},
};
constexpr int NUM_CASES = sizeof(CASES) / sizeof(CASES[0]);

} // namespace

int main() {
    int total_agree = 0;
    for (int c = 0; c < NUM_CASES; ++c) {
        long array_size = CASES[c].array_size;
        long stride = CASES[c].stride;

        int predicted = classify_pathological(array_size, stride);
        int observed = observed_pathological(array_size, stride);
        int agree = (predicted == observed) ? 1 : 0;
        total_agree += agree;

        printf("case=%d array_size=%ld stride=%ld predicted=%d observed=%d agree=%d\n",
               c, array_size, stride, predicted, observed, agree);
    }
    printf("total_agree=%d\n", total_agree);
    return 0;
}
