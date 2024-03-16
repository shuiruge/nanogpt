import tensorflow as tf
from keras.layers import Layer, Dense, LayerNormalization, Embedding


def swapaxes(x, axis1, axis2):
  """
  Examples:
  >>> x = ...  # shape: [..., 2, 3, 4]
  >>> swapaxes(x, -3, -2)  # shape: [..., 3, 2, 4]

  Args:
    x: tf.Tensor
    axis1: int
    axis2: int

  Returns: tf.Tensor
  """
  rank = len(x.get_shape().as_list())
  if axis1 < 0:
    axis1 = rank + axis1
  if axis2 < 0:
    axis2 = rank + axis2

  perm = []
  for axis in range(rank):
    if axis == axis1:
      perm.append(axis2)
    elif axis == axis2:
      perm.append(axis1)
    else:
      perm.append(axis)

  return tf.transpose(x, perm)


def reshape_last_axes(x, shape, num_axes):
  """
  Examples:
  >>> x = ...  # shape: [..., m, n]
  >>> reshape_last_axes(x, [m * n], 2)  # shape [..., (m * n)]
  >>> x = ...  # shape: [..., (m * n)]
  >>> reshape_last_axes(x, [m, n], 1)  # shape [..., m, n]

  Args:
    x: tf.Tensor
    shape: Iterable[int]
    num_axes: int

  Returns: tf.Tensor
  """
  orig_shape = tf.shape(x)
  new_shape = tf.concat(
      [orig_shape[:(-num_axes)], shape], axis=0)
  return tf.reshape(x, new_shape)


def attention(query, key, value, mask=None):
  """Calculates the attention weights. Luong-style attention.

  The query, key, value must have matching leading dimensions. And the key and
  value must have matching penultimate dimension: seq_len_k == seq_len_v.

  The mask has different shapes depending on its type (padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    query: tf.Tensor
      Shape: [..., seq_len_q, depth]
    key: tf.Tensor
      Shape: [..., seq_len_k, depth]
    value: tf.Tensor
      Shape: [..., seq_len_v, depth_v]
    mask: Optional[tf.Tensor]
      Float tensor with shape broadcastable to [..., seq_len_q, seq_len_k].
      The maksed elements are represented by `1` and others by `0`.
      Defaults to all zeros.

  Returns: Tuple[tf.Tensor, tf.Tensor]
    output: shape [..., seq_len_q, depth]
    attention_weights: shape [..., seq_len_q, seq_len_k]
  """
  # Ensure `seq_len_k == seq_len_v`
  # assert tf.shape(key)[-2] == tf.shape(value)[-2]

  # [..., seq_len_q, seq_len_k]
  matmul_qk = tf.matmul(query, key, transpose_b=True)

  # Scale matmul_qk:
  dk = tf.cast(tf.shape(key)[-1], matmul_qk.dtype)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # Add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # Softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  # [..., seq_len_q, seq_len_k]
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

  output = tf.matmul(attention_weights, value)  # [..., seq_len_q, depth_v]
  return output, attention_weights


class MultiHeadWrapper:
  """A wrapper for multi-head operations.

  Examples:
  >>> multi_head = MultiHeadWrapper(8)
  >>> query, key, value = ...
  >>> query = multi_head.split_heads(query)
  >>> ...
  >>> output, weights = attention(query, key, value)
  >>> output = multi_head.concat_heads(output)
  
  Args:
    num_heads: int
  """

  def __init__(self, num_heads):
    self.num_heads = num_heads
  
  def split_heads(self, x):
    """Split the last dimension into (num_heads, depth).

    Args:
      x: tf.Tensor
        Shape [..., seq_len, input_dim].
        Make sure that `input_dim % num_heads == 0`.

    Returns: tf.Tensor
      Shape [..., num_heads, seq_len, depth].
    """
    input_dim = tf.shape(x)[-1]
    # assert input_dim % self.num_heads == 0

    depth = input_dim // self.num_heads
    # [..., seq_len, num_heads, depth]
    x = reshape_last_axes(x, [self.num_heads, depth], 1)
    # [..., num_heads, seq_len, depth]
    x = swapaxes(x, -3, -2)
    return x

  def concat_heads(self, x):
    """Inverse of `self._split_heads`.

    Args:
      x: tf.Tensor
        Shape: [..., num_heads, seq_len, depth].

    Returns: tf.Tensor
      Shape: [..., seq_len, (num_heads * depth)]
    """
    depth = tf.shape(x)[-1]
    # [..., seq_len, num_heads, depth]
    x = swapaxes(x, -3, -2)
    # [..., seq_len, (num_heads * depth)]
    x = reshape_last_axes(x, [self.num_heads * depth], 2)
    return x


class CausalMasker:
  
  def __init__(self, seq_len):
    self.seq_len = seq_len
  
  def __call__(self, x):
    return NotImplemented


class FeedForward(Layer):
  """Feed-forward layers. The hidden layers have ReLU activations. The output
  layer is linear.

  Args:
    hidden_units: List[int]
    output_dim: int
    activition: str or Callable[tf.Tensor, [tf.Tensor]]
  """

  def __init__(self, hidden_units, output_dim, activation='relu', **kwargs):
    super().__init__(**kwargs)
    self.hidden_units = hidden_units
    self.output_dim = output_dim
    self.activation = activation
  
    self.hidden_layers = [Dense(n, activation) for n in hidden_units]
    self.output_layer = Dense(output_dim)
  
  def call(self, x):
    for layer in self.hidden_layers:
      x = layer(x)
    return self.output_layer(x)


class ResBlockWrapper(Layer):
  """A decorator class for ResNet block.

  It can be implemented by closure, like any other decorator. But, closure may
  hide the layer-normalization, which contains information that we may want to
  explore. So, we implement it as a callable class instead.

  Args:
    fn: Callable[tf.Tensor, [tf.Tensor]]
      Input and output shall share the same shape and dtype.
  """

  def __init__(self, fn, **kwargs):
    super().__init__(**kwargs)
    self.fn = fn

    self.layer_norm = LayerNormalization()

  def __call__(self, x):
    return x + self.fn(self.layer_norm(x))


class Embed(Layer):
  """Token embedding + position embedding."""
  
  def __init__(self, vocab_size, max_seq_len, embed_dim, **kwargs):
    super().__init__(**kwargs)
    self.vocab_size = vocab_size
    self.max_seq_len = max_seq_len
    self.embed_dim = embed_dim
  
    self.token_embed = Embedding(vocab_size, embed_dim)
    self.position_embed = Embedding(max_seq_len, embed_dim)
  
  def call(self, x):
    """
    Args:
      x: tf.Tensor
        Shape: [..., seq_len]. Make sure that seq_len < self.max_seq_len.

    Returns: tf.Tensor
      Shape: [..., seq_len, self.embed_dim]
    """
    # x shape: [..., seq_len]
    seq_len = tf.shape(x)[-1]
    # assert seq_len <= self.max_seq_len

    token = self.token_embed(x)  # [..., seq_len, embed_dim]

    # [seq_len, embed_dim]
    position = self.position_embed(tf.range(0, seq_len))
    # [..., seq_len, embed_dim]
    position = tf.broadcast_to(position, tf.shape(token))

    return token + position


class CharacterTokenizer:
  """
  Args:
    corpus: str
  """  
  def __init__(self, corpus):
    self.vocab = sorted(list(set(corpus)))
    self.vocab_size = len(self.vocab)

    self.ctoi = {c: i for i, c in enumerate(self.vocab)}
    self.itoc = {i: c for i, c in enumerate(self.vocab)}

  def encode(self, text):
    return [self.ctoi[c] for c in text]
  
  def decode(self, token_ids):
    return ''.join([self.itoc[i] for i in token_ids])
