#include "sol.hpp"

std::shared_ptr<Payload> make_payload(bool use_make_shared, int a, double b, char c) {
    if (use_make_shared) {
        return std::make_shared<Payload>(Payload{a, b, c});
    }
    return std::shared_ptr<Payload>(new Payload{a, b, c});
}
