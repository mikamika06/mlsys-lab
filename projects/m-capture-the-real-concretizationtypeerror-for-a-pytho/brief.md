# Intermittent failure and tracing semantics in compilation pipeline

We noticed inconsistent behavior during tracing in our compilation pipeline. When developers write control flow using standard Python `if` statements over traced values inside custom tracer wrappers, the system does not fail with the intended `ConcretizationTypeError` exception from JAX-like tracing rules. Instead, errors are either swallowed, converted into generic `TypeError` exceptions, or bypassed entirely. This prevents clear diagnostic reports for upstream ML compiler engineers who rely on precise error typing when improper dynamic control flow occurs.

Furthermore, our internal telemetry for tracking JIT re-compilation events across changing input array shapes is producing inaccurate retrace metrics. Re-tracing is triggered intermittently or counted redundantly, making it impossible to audit recompilation overhead when passing standard primitive types versus structured inputs. Finally, the behavior comparison between treating integer values as static function arguments versus regular array arguments produces inconsistent compilation counters.

To resolve these pipeline issues:
1. Wrap tracer execution so that evaluating Python `if` statements over symbolic/traced boolean array variables accurately catches and propagates a `ConcretizationTypeError` exception containing the underlying traced value information.
2. Implement an execution runner equipped with an internal retrace counter that cleanly records how many times functions are actually re-compiled when called with arrays of 3 distinct shapes.
3. Measure and report the difference in re-compilation behavior when passing integer arguments as `static_argnums` versus standard array arguments across distinct values.
4. Add regression tests in `tests/test_regression.py` validating that dynamic control flow errors are strictly asserted and retrace counters behave deterministically.
