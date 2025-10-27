import jax.numpy as jnp

import jax

print(f"JAX backend: {jax.devices()}")

x = jnp.ones((3, 3))
y = jnp.dot(x, x)
print(y)
print(y.device)
