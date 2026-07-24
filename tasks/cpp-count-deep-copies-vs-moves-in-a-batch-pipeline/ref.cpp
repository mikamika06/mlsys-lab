#include "sol.hpp"
#include <utility>

void run_pipeline(const Op* ops, int n, std::vector<Buffer>& vec) {
    for (int i = 0; i < n; i++) {
        const Op& op = ops[i];
        switch (op.kind) {
            case 0: {  // push_temp: moves a temporary in
                vec.push_back(Buffer());
                break;
            }
            case 1: {  // push_lvalue: copies a named object in
                Buffer b;
                vec.push_back(b);
                break;
            }
            case 2: {  // copy_assign
                vec[op.dst] = vec[op.src];
                break;
            }
            case 3: {  // move_assign
                vec[op.dst] = std::move(vec[op.src]);
                break;
            }
            default:
                break;
        }
    }
}
