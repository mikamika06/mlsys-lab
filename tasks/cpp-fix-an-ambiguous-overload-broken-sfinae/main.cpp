// FIXED driver. Trivial on purpose: the `process` template overloads and
// the classification test that exercises them both have to live in
// whichever .cpp defines them (function templates need their definition
// visible in every translation unit that instantiates them), so all the
// real work happens in ref.cpp / solve.cpp's run_overload_tests().
#include "sol.hpp"

int main() {
    run_overload_tests();
    return 0;
}
