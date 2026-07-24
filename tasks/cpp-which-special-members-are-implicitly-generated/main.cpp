#include <cstdio>
#include "sol.hpp"

int main() {
    // Mirrors 10 real class definitions:
    //   C0: struct C0 {};
    //   C1: struct C1 { ~C1(){} };
    //   C2: struct C2 { C2(const C2&){} };
    //   C3: struct C3 { C3& operator=(const C3&){return *this;} };
    //   C4: struct C4 { C4(C4&&){} };
    //   C5: struct C5 { C5& operator=(C5&&){return *this;} };
    //   C6: struct C6 { C6(){} };
    //   C7: struct C7 { C7(const C7&){} C7& operator=(const C7&){return *this;} };
    //   C8: struct C8 { ~C8(){} C8(C8&&) = default; C8& operator=(C8&&) = default; };
    //   C9: struct C9 { C9() = delete; };
    ClassDecl classes[10] = {
        {0, false, false, false, false, false},  // C0
        {0, true,  false, false, false, false},  // C1
        {0, false, true,  false, false, false},  // C2
        {0, false, false, true,  false, false},  // C3
        {0, false, false, false, true,  false},  // C4
        {0, false, false, false, false, true},   // C5
        {1, false, false, false, false, false},  // C6
        {0, false, true,  true,  false, false},  // C7
        {0, true,  false, false, true,  true},   // C8
        {2, false, false, false, false, false},  // C9
    };

    for (int i = 0; i < 10; i++) {
        MemberAvail m = classify_special_members(classes[i]);
        printf("C%d: %d %d %d %d %d %d\n", i,
               (int)m.default_ctor, (int)m.dtor, (int)m.copy_ctor,
               (int)m.copy_assign, (int)m.move_ctor, (int)m.move_assign);
    }
    return 0;
}
