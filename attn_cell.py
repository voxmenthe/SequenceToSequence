
# we are going to call this attention cell at every time step
# just need to pass in hidden states from the encoder
# the prev_decoder_state is already there in the RNN base class??
class attn_cell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, en_states, input_size=None, activation=tanh):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation
    self._en_states = tf.concat(en_states,0) # just tensorizing
    self._attn_maps = []

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  # calculate context vector c_i
  def get_context(self,_en_states,prev_decoder_state): # A1.2	  

    # score for every input time step
	  scores = _linear(tf.tanh(_linear(prev_decoder_state,self._num_units,bias=False)+
      _linear(_en_states,self._num_units,bias=False),axis=??),1,bias=False)

    scores_exp_sum = tf.reduce_sum(tf.exp(scores),axis=??)
    probs_alpha_ij = tf.exp(scores) / scores_exp_sum
    context_vector_c_i = tf.reduce_sum(probs_alpha_ij * _en_states,axis=0?1?)

    # this can be one of the variables to be returned in session.run()
    self._attn_maps.append(probs_alpha_ij)

	return context_vector_c_i

  # enhanced GRU which needs a context vector c_i which
  # needs to be recalculated at every time step by calling get_context
  # provide history of all hidden states in encoder and the current hidden
  # state in the decoder
  def __call__(self, inputs, prev_decoder_state, scope=None): # A1.1
    """Gated recurrent unit (GRU) with nunits cells."""
    with vs.variable_scope(scope or "gru_cell"):
      with vs.variable_scope("gates"):  # Reset gate and update gate.
        
        # We start with bias of 1.0 to not reset and not update.
        # weight matrix U is hidden in _linear
        # declaring a weight matrix and multiplying by it is combined into
        # function _linear - common convention in TF RNN co

        r, z = array_ops.split(
            value=_linear(
                [inputs, state], 2 * self._num_units, True, 1.0),
            num_or_size_splits=2,
            axis=1)

        context_vector = get_context(_en_states,prev_decoder_state)

        with vs.variable_scope("r"):
          C_c_i_r = _linear(context_vector,bias=False)
        with vs.variable_scope("z"):
          C_c_i_z = _linear(context_vector,bias=False)
        with vs.variable_scope("s"):
          C_c_i_s = _linear(context_vector,bias=False)


        r, z = sigmoid(r+C_c_i_r), sigmoid(z+C_c_i_z)

      with vs.variable_scope("candidate"):
        # this 'c' is s~_i
        s = self._activation(_linear([inputs, r * state],
                                     self._num_units, True) + C_c_i_s)
      # state is s_i-1
      # c is s~_i
      new_h = z * state + (1 - z) * s
    return new_h, new_h


def _linear(args, output_size, bias, bias_start=0.0):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        "weights", [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = vs.get_variable(
          "biases", [output_size],
          dtype=dtype,
          initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
    return nn_ops.bias_add(res, biases)

