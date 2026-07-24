def module_globals_probe():
    namespace = module_globals_probe.__globals__
    same_object = namespace is globals()
    before = namespace["ARENA_GLOBAL"]
    namespace["ARENA_GLOBAL"] = "mutated"
    after = ARENA_GLOBAL
    return same_object, before, after
