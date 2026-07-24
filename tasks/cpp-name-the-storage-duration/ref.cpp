#include "sol.hpp"

// Reference: the correct label for each of the 20 declarations in
// task.md, in order (5 types x {automatic, static, thread, dynamic}).
void name_storage_durations(std::string out[20]) {
    static const char* labels[20] = {
        "automatic", "static", "thread", "dynamic",
        "automatic", "static", "thread", "dynamic",
        "automatic", "static", "thread", "dynamic",
        "automatic", "static", "thread", "dynamic",
        "automatic", "static", "thread", "dynamic",
    };
    for (int i = 0; i < 20; i++) out[i] = labels[i];
}
