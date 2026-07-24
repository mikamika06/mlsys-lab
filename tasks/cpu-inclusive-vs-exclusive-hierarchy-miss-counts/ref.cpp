#include "sol.hpp"
#include <list>
#include <unordered_map>

namespace {

// A fully-associative LRU set of `cap` lines. MRU at the front of the list.
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

    void remove(long line) {
        auto it = map_.find(line);
        if (it == map_.end()) return;
        order_.erase(it->second);
        map_.erase(it);
    }

    // Insert `line` (caller guarantees it is not already present). Returns
    // the evicted line if the set was full, or -1 if nothing was evicted.
    long insert(long line) {
        long evicted = -1;
        if (static_cast<int>(order_.size()) >= cap_) {
            evicted = order_.back();
            order_.pop_back();
            map_.erase(evicted);
        }
        order_.push_front(line);
        map_[line] = order_.begin();
        return evicted;
    }

private:
    int cap_;
    std::list<long> order_;
    std::unordered_map<long, std::list<long>::iterator> map_;
};

long run_inclusive(const long* addrs, int n) {
    LruSet l1(L1_WAYS), l2(L2_WAYS);
    long misses = 0;
    for (int i = 0; i < n; ++i) {
        long line = addrs[i] / LINE_BYTES;
        if (l1.contains(line)) {
            l1.touch(line);
            l2.touch(line); // inclusive: L2 also tracks recency
            continue;
        }
        if (l2.contains(line)) {
            l2.touch(line);
            long evicted_l1 = l1.insert(line);
            (void)evicted_l1; // still resident in L2, no back-invalidation needed
            continue;
        }
        ++misses;
        long evicted_l2 = l2.insert(line);
        if (evicted_l2 != -1 && l1.contains(evicted_l2)) {
            l1.remove(evicted_l2); // back-invalidation
        }
        l1.insert(line);
    }
    return misses;
}

long run_exclusive(const long* addrs, int n) {
    LruSet l1(L1_WAYS), l2(L2_WAYS);
    long misses = 0;
    for (int i = 0; i < n; ++i) {
        long line = addrs[i] / LINE_BYTES;
        if (l1.contains(line)) {
            l1.touch(line);
            continue;
        }
        if (l2.contains(line)) {
            l2.remove(line); // exclusive: promote out of L2
            long evicted_l1 = l1.insert(line);
            if (evicted_l1 != -1) {
                l2.insert(evicted_l1); // victim fill
            }
            continue;
        }
        ++misses;
        long evicted_l1 = l1.insert(line);
        if (evicted_l1 != -1) {
            l2.insert(evicted_l1); // victim fill
        }
    }
    return misses;
}

} // namespace

void hierarchy_miss_counts(const long* addrs, int n, long* out2) {
    out2[0] = run_inclusive(addrs, n);
    out2[1] = run_exclusive(addrs, n);
}
