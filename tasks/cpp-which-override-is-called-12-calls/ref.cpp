#include "sol.hpp"

// Reference / oracle: the tags are NOT hardcoded — they are produced by
// actually executing the 12 calls under real C++ virtual-dispatch rules.
namespace {
struct Base {
    virtual int who() { return 1; }   // Base::who      -> tag 1
    int nonvirt()     { return 10; }  // Base::nonvirt   -> tag 10  (NON-virtual)
    virtual ~Base() {}
};
struct Derived : Base {
    int who() override { return 2; }  // Derived::who    -> tag 2
    int nonvirt()      { return 20; } // Derived::nonvirt -> tag 20 (hides Base::nonvirt)
};
struct MoreDerived : Derived {
    int who() override { return 3; }  // MoreDerived::who -> tag 3
};
}  // namespace

void predict_tags(int out[12]) {
    Base        b;
    Derived     d;
    MoreDerived m;

    Base*    pb_d  = &d;      // static Base*,    dynamic Derived
    Base&    rb_d  = d;       // static Base&,    dynamic Derived
    Base*    pb_b  = &b;      // static Base*,    dynamic Base
    Base*    pb_m  = &m;      // static Base*,    dynamic MoreDerived
    Derived* pd_m  = &m;      // static Derived*, dynamic MoreDerived
    Base     sliced = d;      // object slicing -> a plain Base
    Base*    pb_d2 = &d;      // static Base*,    dynamic Derived (non-virtual call)
    Base&    rb_m  = m;       // static Base&,    dynamic MoreDerived

    out[0]  = b.who();          //  1  object call, static type Base
    out[1]  = d.who();          //  2  object call, static type Derived
    out[2]  = pb_d->who();      //  2  virtual via ptr  -> dynamic Derived
    out[3]  = rb_d.who();       //  2  virtual via ref  -> dynamic Derived
    out[4]  = pb_b->who();      //  1  virtual via ptr  -> dynamic Base
    out[5]  = pb_m->who();      //  3  virtual via ptr  -> dynamic MoreDerived
    out[6]  = pd_m->who();      //  3  virtual via Derived* -> dynamic MoreDerived
    out[7]  = sliced.who();     //  1  sliced object, now a Base
    out[8]  = b.nonvirt();      // 10  non-virtual, static type Base
    out[9]  = d.nonvirt();      // 20  non-virtual, static type Derived
    out[10] = pb_d2->nonvirt(); // 10  non-virtual via Base* -> uses STATIC type Base
    out[11] = rb_m.who();       //  3  virtual via ref  -> dynamic MoreDerived
}
