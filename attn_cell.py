
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


attnCell(self, inputs, prev_decoder_state, scope=None)

