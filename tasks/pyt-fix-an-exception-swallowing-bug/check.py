"""
Real oracle: for each probe function, we directly execute it ourselves
(in our own try/except) to observe what CPython actually raises, then
require the solution to report that exact same observation. Nothing here
is a hardcoded expected string.
"""


class _CustomError(Exception):
    """A normal, learner-code-defined error."""


class _ControlSignal(BaseException):
    """A BaseException that is NOT an Exception -- like SystemExit /
    KeyboardInterrupt, this must propagate through classify_failure
    unchanged, never be reported as a string."""


def _ok():
    return 42


def _value_err():
    raise ValueError("bad value")


def _key_err():
    return {}["missing"]


def _zero_div():
    return 1 / 0


def _type_err():
    return "a" + 1  # noqa: intentional TypeError probe


def _custom_err():
    raise _CustomError("custom")


def _nested_err():
    def inner():
        raise KeyError("deep")

    def middle():
        inner()

    middle()


def _control_signal():
    raise _ControlSignal("stop")


_EXPECT_STRING_CASES = [_ok, _value_err, _key_err, _zero_div, _type_err, _custom_err, _nested_err]


def _direct_outcome(fn):
    """Real oracle: actually run fn() and observe what happens."""
    try:
        fn()
        return "OK"
    except Exception as exc:
        return type(exc).__name__


def grade(sol, fx) -> dict:
    if not hasattr(sol, "classify_failure"):
        return {"exact_match": 0.0}

    ok = 1.0

    for fn in _EXPECT_STRING_CASES:
        expected = _direct_outcome(fn)
        try:
            got = sol.classify_failure(fn)
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break

    if ok == 1.0:
        # A BaseException that is NOT an Exception must propagate through
        # untouched -- classify_failure must not catch (or must re-raise)
        # it, and must never report it as a return-value string.
        try:
            sol.classify_failure(_control_signal)
            ok = 0.0  # swallowed the control signal -- wrong
        except _ControlSignal:
            pass  # correctly propagated
        except Exception:
            ok = 0.0  # propagated, but as the wrong type

    return {"exact_match": ok}
