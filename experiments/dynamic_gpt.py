import sys
sys.path.append('../nanogpt')

import numpy as np
import tensorflow as tf
from keras.layers import Layer, Dense, LayerNormalization, Dropout
from keras.models import Model
from keras.optimizers import AdamW
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import EarlyStopping
from dataclasses import dataclass
from typing import List

# from nanogpt.utils import (
from utils import (
    CharacterTokenizer, LanguageModelDataGenerator, TokPosEmbedding, FeedForward,
    MultiHeadWrapper, ResNetWrapper, luong_attention,
)
from node.core import get_node_function
from node.solvers import runge_kutta as rk

tf.random.set_seed(42)

with open('data/shakespeare.txt', 'r') as f:
    corpus = ''.join(f.readlines())

tokenizer = CharacterTokenizer(corpus)

seq_len = 64
data = LanguageModelDataGenerator(tokenizer.encode(corpus))

contexts = []
targets = []
while True:
    try:
        context, target = data(seq_len, False)
        contexts.append(context)
        targets.append(target)
    except StopIteration:
        break
contexts = np.stack(contexts).astype('int64')
targets = np.asarray(targets, 'int64')


class DynamicGPT(tf.keras.layers.Layer):
  r"""
  Examples:

  ```python
  from node.solvers.runge_kutta import RKF56Solver

  depth = 8
  num_heads = 4
  ffd_hidden_units = [128]
  solver = RKF56Solver(0.01, 1e-3, 1e-2)
  gpt_layer = DynamicGPT(depth, num_heads, ffd_hidden_units, solver, 1.)

  x = tf.random.uniform(shape=[16, 32, depth])
  y = gpt_layer(x)
  print([_.shape for _ in y])
  ```
  """

  def __init__(self, model_dim, num_heads, ffd_hidden_units, solver, t):
    super().__init__()

    attention = VanillaSelfAttention(num_heads)
    feed_forward = FeedForward(ffd_hidden_units, model_dim)

    dynamics = get_albert_dynamics(self_attention, feed_forward)
    t0 = tf.constant(0.)
    t = tf.convert_to_tensor(t)
    signature = [[
        tf.TensorSpec(shape=[None, None, depth], dtype=self.dtype),
        tf.TensorSpec(shape=[None, 1, None, 1], dtype=self.dtype),
        tf.TensorSpec(shape=[None, None, depth], dtype=self.dtype)
    ]]
    node_fn = get_node_function(solver, t0, dynamics, signature=signature)

    def output_fn(x):
      return node_fn(t, x)[0]

    self._self_attention = self_attention
    self._output_fn = output_fn

  def call(self, inputs):
    x, mask = inputs
    att = self._self_attention(x, mask)
    return self._output_fn([x, mask, att])


class VanillaSelfAttention(Layer):
    """A vanilla version of self-attention.

    It is called vanilla because it has no trainable variables at all! The
    function of this kind of self-attention is just communicating between
    each token (or each node if there is an image of community in your mind).
    So, it has no resposibility for computation, which is completely left to
    feed-forward layers.

    For each node i with state $x_i$ and each node j with state $x_j$, the
    the information propagate from node i to node j is $w_{ij} x_i$, where
    $w_{ij}$ is proportional to $\exp(x_i \dot x_j)$ and is normalized so that,
    for all nodes that sends to node j, the total $\sum_i w_{ij}$ shall be unit.

    In this communication viewpoint, the multi-head means the multi-channel
    of communication. Each channel propagates one kind of information. And
    different kinds of information are sent to different target nodes.

    We mask out the self-communication. This is like a Hopfield network, where
    self-interaction is neglected (the weight matrix has a vanished diagonal).

    Args:
        num_heads: int
    """

    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

        self.multihead = MultiHeadWrapper(self.num_heads)

    def call(self, x, return_weights=False):
        """
        Args:
            x: tf.Tensor
                Shape [..., seq_len, dim]
            return_weights: bool
                If return the attention weights. Defaults to False.

        Returns: tf.Tensor or (tf.Tensor, tf.Tensor)
            If return_weights is false, then return the output only, which has
            the same shape and dtype as the x. Otherwise, return the output as
            well as the attention weights, which has shape
            [..., num_heads, seq_len, seq_len].
        """
        # Split the last dimension into multiple heads.
        query = self.multihead.split_heads(x)
        key = self.multihead.split_heads(x)
        value = self.multihead.split_heads(x)

        # Mask the self-communication.
        seq_len = tf.shape(x)[-2]
        mask = tf.linalg.diag(tf.ones([seq_len]))

        # The communication is implemented by a Luong-style attention.
        output, attention_weights = luong_attention(query, key, value, mask)

        # Concatenate the heads together.
        output = self.multihead.concat_heads(output)
        return (output, attention_weights) if return_weights else output


@dataclass
class GPTConfig:
    vocab_size: int
    seq_len: int
    embed_dim: int
    model_dim: int
    num_heads: int
    ffd_hidden_units: List[int]
    T: float
