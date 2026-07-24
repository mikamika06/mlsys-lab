from typing import Any, Callable


def classify_failure(fn: Callable[[], Any]) -> str:
    """
    Call ``fn()`` with no arguments and classify the outcome:

    * If it returns normally, return the string ``"OK"``.
    * If it raises something that IS an instance of ``Exception`` (a normal
      error), catch it and return ``type(exc).__name__`` -- the exact class
      name of whatever was actually raised.
    * If it raises something that is a ``BaseException`` but NOT an
      ``Exception`` (a control-flow signal such as ``SystemExit``,
      ``KeyboardInterrupt``, or a custom ``BaseException`` subclass), do
      NOT catch it -- let it propagate to the caller unchanged.
    """
    try:
        fn()
    except Exception as exc:
        return type(exc).__name__
    return "OK"
