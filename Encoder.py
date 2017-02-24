import tensorflow as tf
import random

from DataUtils import DataUtils
from Config import Config


class Seq2SeqModel(LanguageModel):

    def load_data(self):
        dataset = [[] for _ in self.config.buckets]

        from_train_file = self.config.from_train_file
        to_train_file = self.config.to_train_file

        max_size = self.config.max_size

        with tf.gfile.GFile(from_train_file, "r") as from_file:
            with tf.gfile.GFile(to_train_file, "r") as to_file:
                from_line, to_line = from_file.readline(), to_file.readline()
                while from_line and to_line and (not max_size or count < max_size):
                    count = 0
                    if count % 5000 == 0:
                        print("Line: ", count)
                    from_ids = [int(x) for x in from_line.split()]
                    to_ids = [int(x) for x in to_line.split()]
                    to_ids.append(self.config.EOS_ID)

                    for bucket_id, (from_size, to_size) in enumerate(self.config.buckets):
                        data_set[bucket_id].append([from_ids, to_ids])
                    from_line, to_line = from_line.readline(), to_line.readline()
        return data_set

    def create_feed_dict(self):
        feed_dict = {}

        encoder_inputs = []
        decoder_inputs = []
        to_weights = []

        for i in xrange(self.config.buckets[-1][0]):
            encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                name="encoder{0}".format(i)))
        feed_dict['encoder_inputs'] = encoder_inputs
        for i in xrange(self.config.buckets[-1][1]+1):
            decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                name="decoder{0}".format(i)))
            to_weights.append(tf.placeholder(tf.float32, shape=[None],
                                name="weight{0}".format(i)))
        targets = [decoder_inputs[i+1] for i in xrange(len(decoder_inputs))]

        feed_dict['decoder_inputs'] = decoder_inputs
        feed_dict['to_weights'] = to_weights
        feed_dict['targets'] = targets

        return feed_dict


    def add_placeholders(self):
        self.en_input_placeholder = tf.placeholder(tf.int32, shape=[None])
        self.de_input_placeholder = tf.placeholder(tf.int32, shape=[None])

    def add_projection(self):
        num_samples = self.config.num_samples
        if num_samples > 0 and num_samples < self.config.to_vocab_size:
            self.w_t = tf.get_variable("W", [self.config.to_vocab_size, self.config.size], dtype=self.config.dtype)
            self.w = tf.transpose(w_t)
            self.b = tf.get_variable("b", [self.config.to_vocab_size], dtype=self.config.dtype)
            return (w,b)

    # Embedding and attention function
    def add_embedding(self):
        """
        input:
        inputs: input placeholder in shape [batch_size, input_size]
        output:
        embedded: embedded input in shape[batch_size*rnn_hidden, time_step_size]
        """
        with tf.device('/cpu:0'):
            with tf.variable_scope("embedding") as scope:
                L = tf.get_variable("L",[self.config.from_vocab_size, self.config.encode_hidden_size], initializer = self.config.initializer)
                embeds = tf.nn.embedding_lookup(L, self.en_input_placeholder)
                embedded = [tf.squeeze(x) for x in tf.split(embeds, [tf.ones([self.config.encode_num_steps], tf.int32)], axis=1)]
        return embedded
    
    def LSTM_cell(self):
        self.cell = tf.contrib.rnn.BasicLSTMCell(self.config.encode_hidden_size)
    
    def encoder_layer(self, inputs):
        """
        inputs: embedded encoder inputs
        outputs: a tuple of (outputs, states)
        """
        initial_state = (tf.zeros([self.config.batch_size, self.config.encode_hidden_size]), tf.zeros([self.config.batch_size, self.config.encode_hidden_size]))
        state = initial_state
        cell = self.cell
        outputs = []
        states = []

        for i in xrange(self.config.encode_num_steps):
            output, state = cell(inputs, state)
            inputs = output
            outputs.append(output)
            states.append(state)
            
        return (outputs, states)

        # Simple attention function - please let me know if the inputs/outputs look
      # correct and if they work with your code - still needs work
      def attention_probs(self, en_states, de_inputs, num_attn_layers,size=1500):
          cell = tf.contrib.rnn.GRUCell(size)

          # 'c' vector is same dimension as all our hidden states
          c = xavier_initialization(en_states.shape)

          # get our 'a' coefficients (scalars)
          a = [c.dot(h_i) for h_i in en_states]

          # get our 'b' values
          exps_sum = np.sum([np.exp(a_i) for a_i in a])
          b = [np.exp(a_i)/exps_sum for a_i in a]

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

    # Loss function
    def add_loss_op(self, inputs, labels):
        labels = tf.reshape(labels, [-1,1])
        w_t = tf.cast(self.w_t, self.config.dtype)
        b = tf.cast(self.b, self.config.dtype)
        inputs = tf.cast(inputs, self.config.dtype)

        softmax_loss = tf.cast(
            tf.nn.sampled_softmax_loss(
                weights=w_t,
                biases=b,
                labels=labels,
                inputs=inputs,
                num_sampled=num_samples,
                num_classes=self.config.to_vocab_size),
                self.config.dtype
            )
        return softmax_loss

    # Add model
    def add_model(self, cell, projection):
        encoder_inputs = self.feed_dict['encoder_inputs']
        decoder_inputs = self.feed_dict['decoder_inputs']
        to_weights     = self.feed_dict['to_weights']
        targets        = self.feed_dict['targets']

        self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
            encoder_inputs, decoder_inputs, targets, to_weights,
            self.config.buckets, lambda x, y: add_embedding(x, y, cell, projection, self.config.forward_only),
            softmax_loss_function=add_loss_op
        )

        if self.config.forward_only:
            if projection is not None:
                for b xrange(len(self.config.buckets)):
                    self.outputs[b] = tf.matmul(x, projection[0] + projection[1] for x in self.outputs[b])
        else:
            params = tf.trainable_variables()
            self.gradient_norms = []
            self.updates = []

            self.lr = tf.Variable(float(self.config.lr),
                                trainable=False, dtype=self.config.dtype)
            self.lr_decay_op = self.lr.assign(self.lr*self.config.learning_rate_decay)
            self.global_step = tf.Variable(0, trainable=False)

            optimizer = tf.train.GradientDescentOptimizer(self.lr)

            for b in xrange(len(self.config.buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                self.config.max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(optimier.apply_gradients(
                    zip(clipped_gradients, params),  global_step=self.global_step
                ))
        self.saver = tf.train.saver(tf.global_variables())

    def step(self, session, encoder_inputs, decoder_inputs, targets, to_weights, b):
        input_feed = {}
        for i in xrange(len(encoder_inputs)):
            name = self.feed_dict['encoder_inputs'][i].name
            input_feed[name] = encoder_inputs[i]

        for i in xrange(len(decoder_inputs)):
            de_name = self.feed_dict['decoder_inputs'][i].name
            to_name = self.feed_dict['to_weights'][i].name
            input_feed[de_name] = decoder_inputs[i]
            input_feed[to_name] = to_weights[i]

        to_last = self.feed_dict['decoder_inputs'][len(decoder_inputs)].name
        input_feed[to_last] = np.zeros([self.config.batch_size], dtype=np.int32)

        if self.config.forward_only:
            output_feed = self.losses[b]
            for i in xrange(len(decoder_inputs)):
                output_feed.append(self.outputs[b][i])
        else:
            output_feed = [self.updates[b], self.gradient_norms[b], self.losses[b]]

        outputs = session.run(output_feed, input_feed)
        if self.config.forward_only:
            return None, outputs[0], outputs[1]
        else:
            return outputs[1], outputs[2], None

    def __init__(self, do_decode=False):
        self.config = Config

        data_set = self.load_data()
        self.add_placeholders()
        output_projection = self.add_projection()

        # Create the internal multi-layer cell
        if self.config.num_layers == 1:
            cell = tf.contrib.rnn.BasicLSTMCell(self.config.size)
        else:
            single_cell = tf.contrib.rnn.BasicLSTMCell(self.config.size)
            cell = tf.contrib.rnn.MultiRNNCell([single_cell for _ in self.config.num_layers])

        self.feed_dict = self.create_feed_dict()
        self.add_model(cell, output_projection, do_decode)