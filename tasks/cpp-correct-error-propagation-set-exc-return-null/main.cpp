#include <cstdio>
#include "sol.hpp"

ExcState g_exc;

void set_error(ExcType type, const std::string& msg) {
    g_exc.type = type;
    g_exc.message = msg;
}

static const char* exc_name(ExcType t) {
    switch (t) {
        case ExcType::ZeroDivisionError: return "ZeroDivisionError";
        default: return "None";
    }
}

int main() {
    const int NF = 7;
    double as[NF] = {10.0, 0.0, -7.5, 3.0, 1e200, 0.0, -4.0};
    double bs[NF] = { 2.0, 5.0,  3.0, 0.0, 1e100, 0.0,  0.0};

    for (int i = 0; i < NF; i++) {
        g_exc.type = ExcType::None;
        g_exc.message.clear();

        PyFloatObj* r = safe_divide(as[i], bs[i]);

        if (r == nullptr) {
            printf("call%d: NULL exc=%s msg=%s\n", i, exc_name(g_exc.type), g_exc.message.c_str());
        } else {
            printf("call%d: value=%.6f exc=%s\n", i, r->value, exc_name(g_exc.type));
            delete r;
        }
    }
    return 0;
}
