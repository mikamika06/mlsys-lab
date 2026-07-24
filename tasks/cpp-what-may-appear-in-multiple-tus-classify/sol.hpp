#pragma once
// One-Definition Rule (ODR) classifier.
//
// Every construct in a C++ program is described here by three properties that
// together decide whether it may be *defined* in more than one translation unit
// (TU) of the same program without breaking the ODR:
//
//   kind      : what sort of entity the construct introduces
//   linkage   : the linkage of the construct's name
//   is_inline : 1 if it is declared 'inline' (or is implicitly inline, e.g. a
//               constexpr function), else 0
//
// The ODR says a program must contain exactly ONE definition of any non-inline
// function or variable with external linkage. It explicitly ALLOWS more than one
// definition (one per TU, token-for-token identical) of: class/union types,
// enumeration types, templates, inline functions and inline variables. Entities
// with internal linkage (or no linkage) are a distinct entity in each TU, so
// defining them in every TU is fine too. Type aliases (typedef / using) are not
// definitions of an entity at all and may appear in every TU.

enum Kind {
    KIND_FUNCTION = 0, // a function
    KIND_VARIABLE = 1, // a namespace-scope variable
    KIND_CLASS    = 2, // a class / struct / union type definition
    KIND_ENUM     = 3, // an enumeration type definition
    KIND_ALIAS    = 4, // a typedef / using type alias
    KIND_TEMPLATE = 5, // a function template or class template
};

enum Linkage {
    LINK_EXTERNAL = 0, // external linkage (one entity shared across TUs)
    LINK_INTERNAL = 1, // internal linkage (static, anon namespace, const at namespace scope)
    LINK_NONE     = 2, // no linkage
};

// Return 1 if a construct with these properties MAY be defined in more than one
// translation unit of the same program without violating the ODR, else 0.
int may_appear_in_multiple_tus(int kind, int linkage, int is_inline);
