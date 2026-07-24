from __future__ import annotations


def predict_import_result(modules: dict, entry: str) -> bool:
    """Predict whether `import <entry>` succeeds or raises `ImportError`,
    given a small import graph -- WITHOUT actually executing any code.

    Parameters
    ----------
    modules : dict[str, list[tuple]]
        Maps module name -> its body, as an ordered list of statements
        executed top to bottom. Each statement is one of:

          ("bind", name)
              Defines `name` in this module's namespace (like `name = 1`).

          ("import_module", other)
              `import other`. Never raises by itself: it just binds a
              reference to whatever module object `other` currently is
              in `sys.modules` -- complete or still mid-execution.

          ("from_import", other, name)
              `from other import name`. Looks up `name` in `other`'s
              namespace AT THAT MOMENT. If `other` hasn't been imported
              yet, `other` is imported first (running its statements from
              the top). If `other` is already present in `sys.modules`
              (e.g. we're in the middle of a cycle and `other` is only
              partially executed so far), its CURRENT namespace is used
              as-is -- no re-execution. If `name` isn't in that namespace
              yet, this raises `ImportError`.

        A module is executed at most once: if control returns to a
        module that is already in `sys.modules` (via `import_module` or
        `from_import`), it is NOT re-run from the top -- its namespace
        so far (complete or partial) is reused directly. This is exactly
        how a circular import in real Python can either resolve cleanly
        or blow up with `ImportError: cannot import name ... from
        partially initialized module ...`, depending on statement order.

    entry : str
        The module name that is imported first (as if a script did
        `import <entry>`).

    Returns
    -------
    bool
        True if the import completes without error, False if it raises
        `ImportError`.
    """
    raise NotImplementedError('your code here')
