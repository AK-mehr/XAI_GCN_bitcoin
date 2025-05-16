import tensorflow as tf
from typing import Tuple, List, Union

from .layers import GCNLayer
from .gcn_skip import GCNTwoLayersSkipConnection


class GCNSequential(tf.keras.Model):
    def __init__(self, layers: List[Union[GCNLayer, GCNTwoLayersSkipConnection]]):
        super(GCNSequential, self).__init__()
        self.model_layers = layers

    def call(self, inputs: Tuple[tf.SparseTensor, tf.Tensor], training=None):
        adj, nodes = inputs
        x = nodes
        # print(f"[GCNSequential] Initial input node features: {x.shape}")
        for i, l in enumerate(self.model_layers):
            if isinstance(l, (GCNLayer, GCNTwoLayersSkipConnection)):
                # print(f"[GCNSequential] Layer {i} - {l.__class__.__name__}: Input shape = {x.shape}")
                x = l([adj, x], training=training)
            else:
                # print(f"[GCNSequential] Layer {i} - {l.__class__.__name__} (non-GCN): Input shape = {x.shape}")
                x = l(x, training=training)
            # print(f"[GCNSequential] Output shape after layer {i}: {x.shape}")
        return x
