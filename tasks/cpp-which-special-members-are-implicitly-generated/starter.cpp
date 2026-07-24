#include "sol.hpp"

// TODO: implement the Rule-of-Five logic from sol.hpp. Right now every
// member is reported "available", which is wrong for every class that
// actually suppresses or deletes something.
MemberAvail classify_special_members(const ClassDecl& d) {
    (void)d;
    MemberAvail m{true, true, true, true, true, true};  // your code here
    return m;
}
