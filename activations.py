import tensorflow as tf


def mish(x):
    """Mish activation function.

    It is defined as:

    ```python
    def mish(x):
        return x * tanh(softplus(x))
    ```

    where `softplus` is defined as:

    ```python
    def softplus(x):
        return log(exp(x) + 1)
    ```

    Example:

    >>> a = tf.constant([-3.0, -1.0, 0.0, 1.0], dtype = tf.float32)
    >>> b = tf.keras.activations.mish(a)
    >>> b.numpy()
    array([-0.14564745, -0.30340144,  0.,  0.86509836], dtype=float32)

    Args:
        x: Input tensor.

    Returns:
        The mish activation.

    Reference:
        - [Mish: A Self Regularized Non-Monotonic
        Activation Function](https://arxiv.org/abs/1908.08681)
    """
    return x * tf.math.tanh(tf.math.softplus(x))

def leaky_relu(x, alpha=0.01):
    """Leaky version of a Rectified Linear Unit.

    It allows a small gradient when the unit is not active:

    `f(x) = alpha * x for x < 0`,
    `f(x) = x for x >= 0`.

    Example:

    >>> a = tf.constant([-3.0, -1.0, 0.0, 1.0], dtype = tf.float32)
    >>> b = tf.keras.activations.leaky_relu(a)
    >>> b.numpy()
    array([-0.03, -0.01,  0.  ,  1.  ], dtype=float32)

    Args:
        x: Input tensor.

    Returns:
        The LeakyReLU activation: `alpha * x for x < 0`, `x for x >= 0`.
    """
    return tf.keras.activations.relu(x, alpha=alpha)

def relu(x):
    """Rectified Linear Unit."""
    return tf.keras.activations.relu(x)

def gelu(x):
    """Gaussian error linear unit (GELU) activation function

    Args:
        x: Input tensor.

    Returns:
        The GELU activation
    """
    return tf.keras.activations.gelu(x)
