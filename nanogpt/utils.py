import numpy as np
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


def luong_attention(query, key, value, mask=None):
  """The Luong-style attention, a fast attention implementation.

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
  """A wrapper for multi-head operations. It splits and concatenates heads.

  Examples:
  >>> multi_head = MultiHeadWrapper(8)
  >>> query, key, value = ...
  >>> query = multi_head.split_heads(query)
  >>> ...  # the same goes for key and value.
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


class MultiHeadSelfAttention(Layer):
  """TODO"""

  def __init__(self, dim, num_heads, **kwargs):
    super().__init__(**kwargs)
    self.dim = dim
    self.num_heads = num_heads

    self.input_linear = Dense(dim, use_bias=False)
    self.output_linear = Dense(dim, use_bias=False)
    self.multi_heads = MultiHeadWrapper(num_heads)
  
  def call(self, x, mask=None, return_weights=False):
    query = self.input_linear(x)
    key = self.input_linear(x)
    value = self.output_linear(x)

    query = self.multi_heads.split_heads(query)
    key = self.multi_heads.split_heads(key)
    value = self.multi_heads.split_heads(value)

    if mask is None:
      # Mask the self-communication.
      seq_len = tf.shape(x)[-2]
      mask = tf.linalg.diag(tf.ones([seq_len]))

    output, attention_weights = luong_attention(query, key, value, mask)
    output = self.multi_heads.concat_heads(output)
    return (output, attention_weights) if return_weights else output


class FeedForward(Layer):
  """Feed-forward layers.
  
  The hidden layers have ReLU activations. The output layer is linear.

  Args:
    hidden_units: List[int]
    output_dim: int
    activation: str or Callable[tf.Tensor, [tf.Tensor]]
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
    x = self.output_layer(x)
    return x


class ResNetWrapper(Layer):
  """x -> x + f(layer_norm(x)), for module f.
  
  Args:
    module: tf.Module
      Both input and output are tf.Tensor. They share the same shape and dtype.
    TODO
  """
  
  def __init__(self, module, input_arg=0, output_arg=0,
               name='resnet_wrapper', **kwargs):
    name += '_of_' + module.name
    super().__init__(name=name, **kwargs)
    self.module = module
    self.input_arg = input_arg
    self.output_arg = output_arg

    self.layer_norm = LayerNormalization()

  def call(self, *args, **kwargs):
    if self.input_arg + 1 > len(args):
      raise ValueError()
    x = args[self.input_arg]

    normed_args = list(args)
    normed_args[self.input_arg] = self.layer_norm(x)

    outputs = self.module(*normed_args, **kwargs)

    if not isinstance(outputs, (list, tuple)):
      # Thus, single output.
      if self.output_arg != 0:
        raise ValueError()
      return x + outputs
    elif self.output_arg + 1 > len(outputs):
      raise ValueError()
    else:  # multiple outputs.
      res_outputs = list(outputs)
      res_outputs[self.output_arg] = x + outputs[self.output_arg]
      return tuple(res_outputs)


class TokPosEmbedding(Layer):
  """Token embedding + position embedding.
  
  The input sequence of token-IDs can be viewed as a sequence of two integers,
  one for the token-ID itself, and one for its position. For example, the input
  [23, 51, 11] can be viewed as [(23, 0), (51, 1), (11, 2)], where the second
  integer in the pairs represents position.
  """
  
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
  """A simple character level tokenizer. It tokenize a text by its characters.
  For example, the text 'accb' is tokenized as ['a', 'c', 'c', 'b'], and then
  converted to token-IDs, like [1, 3, 3, 2].

  Args:
    corpus: str
      The corpus for generating vocabulary.
    placeholders: List[str]
      The placeholders for, e.g. masking, padding, etc. These placeholders are
      appended to the vocabulary. For example, you may use placeholder 'MASK'
      for masking. Defaults to None.
  """  
  def __init__(self, corpus, placeholders=None):
    self.corpus = corpus
    self.placeholders = placeholders
    
    self.vocab = sorted(list(set(corpus)))
    if placeholders is not None:
      # Append the placeholders to the tail, without changing the order of
      # elements in placeholders.
      self.vocab += placeholders
    self.vocab_size = len(self.vocab)

    self.t2i = {t: i for i, t in enumerate(self.vocab)}
    self.i2t = {i: t for i, t in enumerate(self.vocab)}

  def encode(self, text):
    """Convert the text to token-IDs.

    Args:
      text: str

    Returns: List[int]
    """
    return [self.get_id(c) for c in text]
  
  def decode(self, token_ids):
    """Inverse of encode.

    Args:
      token_ids: List[int]

    Returns: str
    """
    return ''.join([self.get_token(i) for i in token_ids])
  
  def get_id(self, token):
    try:
      return self.t2i[token]
    except KeyError:
      raise ValueError(f'Token "{token}" is out of vocabulary.')
  
  def get_token(self, id):
    try:
      return self.i2t[id]
    except KeyError:
      raise ValueError(f'ID {id} is out of vocabulary size.')

 
class LanguageModelDataGenerator:
  """Generate data for language model.

  The task of a language model is to predict the next token (character, word,
  or word-piece) based on the previous tokens.

  Args:
    token_ids: List[int]
  """
  
  def __init__(self, token_ids):
    self.token_ids = np.asarray(token_ids, dtype='int64')
    self.offset = 0

  def __call__(self, seq_len, auto_reset):
    """Generates a context and a target.

    Args:
      seq_len: int
        The length of the context, as a sequence of token-IDs.
      auto_reset: bool
    
    Returns: (List[int], int)

    Raises:
      StopIteration: When `auto_reset = False` and all token-IDs has been
        emitted.
    """
    target_id = self.offset + seq_len

    if target_id >= len(self.token_ids):
      if auto_reset:
        self.offset = 0
        target_id = seq_len
      else:
        raise StopIteration()

    contexts = self.token_ids[self.offset:target_id]
    target = self.token_ids[target_id]
    self.offset += 1
    return contexts, target


class MaskedLanguageModelDataGenerator:
  """Generate data for masked language model.

  TODO

  Args:
    token_ids: List[int]
    mask_token_id: int
    get_mask_positions: Callable: List[int] -> List[int]
  """
  
  def __init__(self, token_ids, mask_token_id, get_mask_positions):
    self.token_ids = np.asarray(token_ids, dtype='int64')
    self.mask_token_id = mask_token_id
    self.get_mask_positions = get_mask_positions

    self.offset = 0

  def __call__(self, seq_len, auto_reset):
    """Generates a context and a target.

    Args:
      seq_len: int
        The length of the context, as a sequence of token-IDs.
      auto_reset: bool
    
    Returns: (List[int], List[int])

    Raises:
      StopIteration: When `auto_reset = False` and all token-IDs has been
        emitted.
    """
    end = self.offset + seq_len

    if end > len(self.token_ids):
      if auto_reset:
        self.offset = 0
        end = seq_len
      else:
        raise StopIteration()

    # Notice that slicing does not copy the data, we have copy manually.
    context = np.copy(self.token_ids[self.offset:end])
    mask = self.get_mask_positions(context)
    target = context[mask]
    context[mask] = self.mask_token_id
    self.offset += 1
    return context, target, mask
