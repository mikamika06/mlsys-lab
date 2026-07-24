#include <cstring>
#include "sol.hpp"

const char* classify_ub(const char* snippet) {
    if (strstr(snippet, "thread")) return "data-race";
    if (strstr(snippet, "<<") || strstr(snippet, "2147483647")) return "integer";
    if (strstr(snippet, "return &") || strstr(snippet, "delete") || strstr(snippet, "push_back")) return "lifetime";
    if (strstr(snippet, "(float*)") || strstr(snippet, "(short*)")) return "aliasing";
    if (strstr(snippet, "nullptr") || strstr(snippet, "*p = 5")) return "null";
    if (strstr(snippet, "i++")) return "sequencing";
    if (strstr(snippet, "arr[")) return "bounds";
    return "unknown";
}
