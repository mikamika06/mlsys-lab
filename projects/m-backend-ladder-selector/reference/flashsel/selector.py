from flashsel.probe import probe_backend

def select_backend(ladder, preferences=None):
    prefs = preferences or []
    ordered = [b for b in prefs if b in ladder] + [b for b in ladder if b not in (preferences or [])]
    for backend in ordered:
        if backend in ladder and probe_backend(backend):
            return backend
    return None
