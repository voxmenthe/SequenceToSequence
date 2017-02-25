# Attention Mechanism From Scratch

# Input: Encoder Hidden States
# Output: Weights For Each Encoder State representing 

# Question: How is it differentiable?
# Question: How does it work together during the training loop? (specifically with TF)
# How does it backprop? (specifically with TF)
# How does it fit in the context of the overall model?

def generateAttentionProbs(encoder_states, decoder_input,num_layers):

    # Output i is computed from input i (which is either the i-th element
    #  of decoder_inputs or loop_function(output {i-1}, i)) as follows.
 
	# First, we run the cell on a combination of the input and previous
	# attention masks:
	#   cell_output, new_state = cell(linear(input, prev_attn), prev_state).
	# 
	# Then, we calculate new attention masks:
	#   new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
	# and then we calculate the output:
	#   output = linear(cell_output, new_attn).

		# LSTM cell?? how does this come in?
		    # Create the internal multi-layer cell for our RNN.
    def single_cell():
      return tf.contrib.rnn.GRUCell(size)
    if use_lstm:
      def single_cell():
        return tf.contrib.rnn.BasicLSTMCell(size)
    cell = single_cell()
    if num_layers > 1:
      cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])


      """Put attention masks on hidden using hidden_features and query."""
  
          # Attention mask is a softmax of v^T * tanh(...).
          s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y),
                                  [2, 3])
          a = nn_ops.softmax(s)
          # Now calculate the attention-weighted vector d.
          ## a is the softmax
          ## applying a reduce_sum with 
          d = math_ops.reduce_sum(
              array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
          ds.append(array_ops.reshape(d, [-1, attn_size]))



	# run softmax

	# return weighted probs associated with each encoder hidden state?

	return 

""" We want to return: 
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors of
        shape [batch_size x output_size]. These represent the generated outputs.
      state: The state of each decoder cell the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

"""
######################################

# we are going to call this attention cell at every time step
class attn_cell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, en_states, input_size=None, activation=tanh):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation
    self._en_states = en_states

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  # calculate context vector c_i
  def get_context(self,_en_states,prev_decoder_state): # A1.2

	  # get our 'a' coefficients (scalars)
	  #score = [c.dot(h_i) for h_i in en_states]
	  weighted_state = _linear(state,_en_states[0].shape)
	  
	  v_a, W_a = _linear(????)
	  
	  score_e_ij = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y),
                                  [2, 3])

	  # get our 'b' values
	  exps_sum = np.sum([np.exp(a_i) for a_i in score])
	  b = [np.exp(a_i)/exps_sum for a_i in score]

	  # then take the b's and multiply by the hidden states to
	  # get output to decoder - need to recalculate for each decoder step?
	  weighted_h_for_decoder = np.sum(tf.mul(b,en_states))

	return c_i

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

        r, u = array_ops.split(
            value=_linear(
                [inputs, state], 2 * self._num_units, True, 1.0),
            num_or_size_splits=2,
            axis=1)

        context_vector = get_context()
        C_c_i = _linear(context_vector)

        r, u = sigmoid(r+C_c_i), sigmoid(u+C_c_i)

      with vs.variable_scope("candidate"):
        # this 'c' is s~_i
        c = self._activation(_linear([inputs, r * state],
                                     self._num_units, True))
      # state is s_i-1
      # c is s~_i
      new_h = u * state + (1 - u) * c
    return new_h, new_h

######################################


      def attention_probs(self, en_states, de_inputs, num_attn_layers,size=1500):
          
          class attn_cell(tf.contrib.rnn.GRUCell):


          # 'c' vector is same dimension as all our hidden states
          c = xavier_initialization(en_states.shape)

          # a and b are our learned parameters?

          # get our 'a' coefficients (scalars)
          score = [c.dot(h_i) for h_i in en_states]

          # get our 'b' values
          exps_sum = np.sum([np.exp(a_i) for a_i in score])
          b = [np.exp(a_i)/exps_sum for a_i in score]

          # then take the b's and multiply by the hidden states to
          # get output to decoder - need to recalculate for each decoder step?
          weighted_h_for_decoder = np.sum(tf.mul(b,en_states))

          # How does the LSTM/GRU cell fit in here? where are the learned params?

          # Local predictive alignment

          # Get attention masks using en_states

          # Alternative implementation using TF ?
          # Attention mask is a softmax of v^T * tanh(...).
          s = math_ops.reduce_sum(v[a] * math_ops.tanh(en_states[a] + y),
                                [2, 3])
          a = nn_ops.softmax(s)

          """ May eventually want to return
          something like:
          A tuple of the form (outputs, state), where:
              outputs: A list of the same length as decoder_inputs of 2D Tensors of
                  shape [batch_size x output_size]. These represent the generated outputs.
              state: The state of each decoder cell the final time-step.
                  It is a 2D Tensor of shape [batch_size x cell.state_size].
          """
          return weighted_h_for_decoder


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


