import sys
import jax
import jax.numpy as jnp

sys.path.insert(0, ".")
from jaxpr_tools.analyzer import count_equations, list_unique_primitives
from jaxpr_tools.tracer import run_with_mutable_closure


def test_count_equations_basic():
    f = lambda x: jnp.sin(x) * 2.0
    _, jaxpr = jax.make_jaxpr(f)(1.0)
    assert count_equations(jaxpr) > 0


def test_list_unique_primitives_basic():
    f = lambda x: jnp.sin(x) + jnp.cos(x)
    _, jaxpr = jax.make_jaxpr(f)(1.0)
    prims = list_unique_primitives(jaxpr)
    assert isinstance(prims, list)
    assert len(prims) > 0


def test_tracer_behavior():
    f = lambda x: x * 3.0
    status, _ = run_with_mutable_closure(f, 2.0)
    assert status in ("leaked", "safe")
