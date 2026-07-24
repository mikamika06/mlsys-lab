#include "sol.hpp"

// TODO: for each op, perform exactly the operation named in sol.hpp:
//   0 push_temp    -> vec.push_back(Buffer());
//   1 push_lvalue  -> Buffer b; vec.push_back(b);
//   2 copy_assign  -> vec[op.dst] = vec[op.src];
//   3 move_assign  -> vec[op.dst] = std::move(vec[op.src]);
void run_pipeline(const Op* ops, int n, std::vector<Buffer>& vec) {
    (void)ops; (void)n; (void)vec;
    // your code here
}
