#include "sol.hpp"

// TODO: implement the rules from sol.hpp. Right now every scenario is
// classified "still valid" (1), which is wrong for every scenario that
// should actually invalidate the iterator.
int classify_iterator_validity(const IterScenario& s) {
    (void)s;
    return 1;  // your code here
}
