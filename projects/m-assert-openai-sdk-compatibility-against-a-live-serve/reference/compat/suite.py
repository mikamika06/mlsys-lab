"""Compatibility execution suite against live server."""

from compat.adapter import transform_request, validate_response


def run_compatibility_suite(server, shapes, make_sample_request_fn):
    results = {}
    for shape in shapes:
        raw_req = make_sample_request_fn(shape)
        transformed = transform_request(shape, raw_req)
        resp = server.handle(transformed)
        is_valid = validate_response(shape, resp)
        results[shape] = {
            "request": transformed,
            "response": resp,
            "valid": is_valid
        }
    return results
