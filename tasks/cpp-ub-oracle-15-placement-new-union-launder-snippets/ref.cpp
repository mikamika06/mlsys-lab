#include "sol.hpp"

int classify_ub(const Op* ops, int n) {
    bool alive = false;
    bool cur_const = false;
    bool cur_trivial = true;
    int cur_type = -1;     // -1: no object has ever occupied this storage
    bool stale = false;    // true: the original pointer needs laundering

    for (int i = 0; i < n; i++) {
        const Op& op = ops[i];
        switch (op.kind) {
            case ALLOCATE:
                alive = false;
                cur_type = -1;
                stale = false;
                break;

            case PLACEMENT_NEW:
                if (alive && !cur_trivial) return 1;             // missing dtor before reuse
                if (cur_type != -1 && cur_const) stale = true;    // reused const-object storage
                alive = true;
                cur_type = op.type;
                cur_const = (op.is_const != 0);
                cur_trivial = (op.is_trivial != 0);
                break;

            case DTOR:
                if (!alive) return 1;                             // double-destroy
                alive = false;
                break;

            case ACCESS:
                if (!alive) return 1;                              // out of lifetime
                if (op.type != cur_type) return 1;                  // wrong active type
                if (op.is_write && cur_const) return 1;              // write through const
                if (stale && !op.laundered) return 1;                 // stale, not laundered
                if (op.laundered) stale = false;                       // fixes it going forward
                break;
        }
    }
    return 0;
}
