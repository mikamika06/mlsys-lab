#include <cstdio>
#include "sol.hpp"

static void reset(Record& r) {
    r.type = 1;
    r.score = 3.14;
    r.flags = 42;
    r.id = 1000;
}

static void printRecord(const char* tag, bool threw, const Record& r) {
    printf("%s %d %d %.10f %d %ld\n", tag, threw ? 1 : 0, (int)r.type, r.score, r.flags, r.id);
}

int main() {
    Op ops[4] = {
        {Field::Type, 2.0},
        {Field::Score, 9.99},
        {Field::Flags, 13.0},
        {Field::Id, 2000.0},
    };

    int throwPoints[5] = {-1, 0, 1, 2, 3};
    for (int k = 0; k < 5; k++) {
        int throwAt = throwPoints[k];
        Record rec;
        reset(rec);
        bool threw = false;
        try {
            transactional_update(&rec, ops, 4, throwAt);
        } catch (const TxnAbort&) {
            threw = true;
        }
        char tag[32];
        snprintf(tag, sizeof(tag), "throw_at_%d", throwAt);
        printRecord(tag, threw, rec);
    }
    return 0;
}
