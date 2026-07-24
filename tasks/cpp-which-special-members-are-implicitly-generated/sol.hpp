#pragma once

// Symbolic description of which special members a class explicitly
// declares (normally, `= default`, or `= delete` -- all count as
// "user-declared" for these rules).
//
// default_ctor: 0 = not user-declared, 1 = user-declared (normal or
// `= default`), 2 = user-declared `= delete`.
// The other five: true if the user wrote that member themselves.
struct ClassDecl {
    int default_ctor;
    bool dtor;
    bool copy_ctor;
    bool copy_assign;
    bool move_ctor;
    bool move_assign;
};

// Result: true if that special member ends up CALLABLE on the class
// (implicitly generated, user-provided, or user-defaulted); false if it
// is absent or implicitly deleted.
struct MemberAvail {
    bool default_ctor;
    bool dtor;
    bool copy_ctor;
    bool copy_assign;
    bool move_ctor;
    bool move_assign;
};

// Apply the real C++11+ "Rule of Five" implicit-special-member rules to
// `d` and return which of the six special members end up available:
//
//   default ctor: unavailable if user-deleted (default_ctor == 2).
//     Otherwise available if user-declared it themselves (default_ctor
//     == 1), OR if the class declares no OTHER constructor at all --
//     i.e. no user copy_ctor and no user move_ctor (copy_assign /
//     move_assign are not constructors and never suppress this).
//
//   destructor: always available for every class in this task.
//
//   copy ctor: available if user-declared it themselves, OR if the class
//     has no user-declared move_ctor and no user-declared move_assign
//     (a user move operation implicitly DELETES the copy ctor).
//   copy assign: the same rule, mirrored.
//
//   move ctor: available if user-declared it themselves, OR if the class
//     has none of: user dtor, user copy_ctor, user copy_assign, user
//     move_assign (any of those simply means no implicit move ctor is
//     generated at all -- not deleted, just absent).
//   move assign: the same rule, mirrored (checking move_ctor instead of
//     move_assign).
MemberAvail classify_special_members(const ClassDecl& d);
