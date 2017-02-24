# Attention Mechanism From Scratch

# Input: Encoder Hidden States
# Output: Weights For Each Encoder State representing 

# Question: How is it differentiable?
# Question: How does it work together during the training loop? (specifically with TF)
# How does it backprop? (specifically with TF)
# How does it fit in the context of the overall model?

def generateAttentionProbs(encoder_states, decoder_input,num_layers,size):

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





