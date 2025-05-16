import tensorflow as tf
from typing import Tuple


class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, units: int, activation=None, kernel_initializer="glorot_uniform", dtype=tf.float32):
        super(GCNLayer, self).__init__(dtype=dtype)
        self.units = int(units)
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)

    def build(self, input_shape):
        last_dim = tf.TensorShape(input_shape[1])[-1]
        self.kernel = self.add_weight(
            name='kernel',
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)
        self.built = True
        # print(f"[GCNLayer] Built layer with input dim {last_dim} and output dim {self.units}")

    def call(self, inputs: Tuple[tf.SparseTensor, tf.Tensor], training=None, mask=None):
        adj, nodes = inputs
        # print(f"[GCNLayer] Input nodes shape: {nodes.shape}")
        x = tf.sparse.sparse_dense_matmul(adj, nodes)
        # print(f"[GCNLayer] After sparse matmul: {x.shape}")
        x = tf.matmul(x, self.kernel)
        # print(f"[GCNLayer] After kernel matmul: {x.shape}")
        if self.activation:
            x = self.activation(x)
            # print(f"[GCNLayer] After activation: {x.shape}")
        return x
