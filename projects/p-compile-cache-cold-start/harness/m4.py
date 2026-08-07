import ref

def check(workdir):
    from compcache.server import Server
    from compcache.warmup import warmup
    s = Server()
    warmup(s, [1, 2])
    warmed = s.is_warmed()
    return ref.get_oracle_warmup(warmed)
