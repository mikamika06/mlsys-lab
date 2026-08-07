import ref

def check(workdir):
    from compcache.server import Server
    from compcache.cache import CompilationCache
    c = CompilationCache()
    c.store("req_shape_1", b"bin")
    s = Server(cache=c)
    cost = s.handle_first_request("req_shape_1")
    return ref.get_oracle_request_cost(cost)
