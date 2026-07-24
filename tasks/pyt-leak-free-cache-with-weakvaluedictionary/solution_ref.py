import gc
import weakref


def cache_surviving_keys(keep):
    class Value:
        pass

    cache = weakref.WeakValueDictionary()
    refs = {}

    for key in range(10):
        value = Value()
        cache[key] = value
        refs[key] = value

    for key in list(refs):
        if key not in keep:
            del refs[key]

    gc.collect()
    return sorted(cache.keys())
