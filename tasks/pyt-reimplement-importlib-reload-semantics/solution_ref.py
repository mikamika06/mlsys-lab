def reload_module_semantics(module, source):
    before = id(module)
    exec(source, module.__dict__)
    return (id(module) == before, module.value)
