import jax
import jax.numpy as jnp


def make_sample_jaxpr():
    f = lambda x: jnp.sin(x) + x * 2.0
    _, jaxpr = jax.make_jaxpr(jax.jit(jax.grad(f)))(1.0)
    return jaxpr


def sample_function(x):
    return x ** 2 + x
