#include <cstdio>
#include "sol.hpp"

static void printVec(const char* tag, const std::vector<Record>& v) {
    printf("%s size=%zu", tag, v.size());
    for (const auto& r : v) printf(" (%d,%.4f)", r.key, r.val);
    printf("\n");
}

int main() {
    // scenario 1: a run of three consecutive matches, plus one more later
    {
        std::vector<Record> v = {
            {1, 1.1}, {2, 2.2}, {2, 2.3}, {2, 2.4}, {3, 3.3}, {2, 2.5}, {4, 4.4},
        };
        eraseByKey(v, 2);
        printVec("scenario1", v);
    }

    // scenario 2: a run of matches right at the start
    {
        std::vector<Record> v = {
            {5, 0.1}, {5, 0.2}, {6, 0.3}, {7, 0.4},
        };
        eraseByKey(v, 5);
        printVec("scenario2", v);
    }

    // scenario 3: no matches at all -- vector must come out unchanged
    {
        std::vector<Record> v = {
            {9, 1.0}, {8, 2.0}, {7, 3.0},
        };
        eraseByKey(v, 100);
        printVec("scenario3", v);
    }

    // scenario 4: every element matches -- result must be empty
    {
        std::vector<Record> v = {
            {1, 1.0}, {1, 2.0}, {1, 3.0},
        };
        eraseByKey(v, 1);
        printVec("scenario4", v);
    }

    return 0;
}
