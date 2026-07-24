#include "sol.hpp"

MemberAvail classify_special_members(const ClassDecl& d) {
    MemberAvail m;

    if (d.default_ctor == 2) {
        m.default_ctor = false;
    } else if (d.default_ctor == 1) {
        m.default_ctor = true;
    } else {
        m.default_ctor = !(d.copy_ctor || d.move_ctor);
    }

    m.dtor = true;

    m.copy_ctor = d.copy_ctor || !(d.move_ctor || d.move_assign);
    m.copy_assign = d.copy_assign || !(d.move_ctor || d.move_assign);

    m.move_ctor = d.move_ctor || !(d.dtor || d.copy_ctor || d.copy_assign || d.move_assign);
    m.move_assign = d.move_assign || !(d.dtor || d.copy_ctor || d.copy_assign || d.move_ctor);

    return m;
}
