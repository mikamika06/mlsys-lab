#include "sol.hpp"

// TODO: when use_make_shared is true, construct via std::make_shared<Payload>
// (one allocation); when false, construct via std::shared_ptr<Payload>(new
// Payload{...}) (two allocations). Either way, fill the Payload with a, b, c.
std::shared_ptr<Payload> make_payload(bool use_make_shared, int a, double b, char c) {
    (void)use_make_shared; (void)a; (void)b; (void)c;
    // your code here
    return std::shared_ptr<Payload>(new Payload{});
}
