import tensorflow as tf
import random

from DataUtils import DataUtils
from Config import Config
from attn_cell import attn_cell, _linear

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
        encoder_inputs = self.bucketize_encoding_layer()
        feed_dict['encoder_inputs'] = encoder_inputs

        decoder_inputs, to_weights, targets = self.bucketize_decoding_layer()

        feed_dict['decoder_inputs'] = decoder_inputs
        feed_dict['to_weights'] = to_weights
        feed_dict['targets'] = targets

        return feed_dict

    def bucketize_encoding_layer():
        encoder_inputs = []

        for i in xrange(self.config.buckets[-1][0]):
            encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                name="encoder{0}".format(i)))
        return encoder_inputs

    def bucketize_decoding_layer():
        decoder_inputs = []
        to_weights = []

        for i in xrange(self.config.buckets[-1][1]+1):
            decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                name="decoder{0}".format(i)))
            to_weights.append(tf.placeholder(tf.float32, shape=[None],
                                name="weight{0}".format(i)))
        targets = [decoder_inputs[i+1] for i in xrange(len(decoder_inputs))]

        return (decoder_inputs, to_weights, targets)

    def get_batch(self, data, bucket_id):
        """Get a random batch of data from the specified bucket, prepare for step.
        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.
        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.
          bucket_id: integer, which bucket to get the batch for.
        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...) later.
        """
        encoder_size, decoder_size = self.config.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in xrange(self.config.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])

            # Encoder inputs are padded and then reversed.
            encoder_pad = [self.config.PAD_ID] * (self.config.encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([self.config.GO_ID] + decoder_input +
                                [self.config.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(self.config.encoder_size):
            batch_encoder_inputs.append(
              np.array([encoder_inputs[batch_idx][length_idx]
                        for batch_idx in xrange(self.config.batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(self.config.decoder_size):
            batch_decoder_inputs.append(
              np.array([decoder_inputs[batch_idx][length_idx]
                        for batch_idx in xrange(self.config.batch_size)], dtype=np.int32))

        # Create target_weights to be 0 for targets that are padding.
        batch_weight = np.ones(self.config.batch_size, dtype=np.float32)
        for batch_idx in xrange(self.config.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
            if length_idx < decoder_size - 1:
                target = decoder_inputs[batch_idx][length_idx + 1]
            if length_idx == decoder_size - 1 or target == self.config.PAD_ID:
                batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights


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
                embeds = tf.nn.embedding_lookup(L, self.feed_dict['encoder_inputs'])
                embedded = [tf.squeeze(x) for x in tf.split(embeds, [tf.ones([self.config.encode_num_steps], tf.int32)], axis=1)]
        return embedded

    def add_decode_embedding(self):
        with tf.variable_scope("decode_embedding") as decode_scope:
            L = tf.get_variable("L", [self.config.to_vocab_size, self.decode_hidden_size], initializer=self.config.initializer)
            embeds = tf.nn.embedding_lookup(L, self.feed_dict['decoder_inputs'])
            embedded = [tf.squeeze(x) for x in tf.split(embeds, [tf.ones([self.config.decode_num_steps], tf.int32)], axis=1)]
        return embedded


    def LSTM_cell(self):
        self.encoder_cell = tf.contrib.rnn.BasicLSTMCell(self.config.encode_hidden_size)
        self.decoder_cell = tf.contrib.rnn.attn_cell(self.config.decode_hidden_size,de_states)

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

    def decoder_layer(self, inputs):
        """
        inputs: embedded encoder inputs
        outputs: a tuple of (outputs, states)
        """
        initial_state = (tf.zeros([self.config.batch_size, self.config.decode_hidden_size]), tf.zeros([self.config.batch_size, self.config.decode_hidden_size]))
        state = initial_state
        cell = self.decoder_cell
        outputs = []
        states = []

        for i in xrange(self.config.decode_num_steps):
            output, state = cell(inputs, state)
            inputs = output
            outputs.append(output)
            states.append(state)

        return (outputs, states)

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

    def add_embeddings(self, encoder_inputs, decoder_inputs, cell, is_decode):
        embeddings = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
            encoder_inputs, decoder_inputs, cell,
            num_encoder_symbols=self.config.from_vocab_size,
            num_decoder_symbols=self.config.to_vocab_size,
            embedding_size=self.config.size,
            output_projection=projection,
            feed_previous=is_decode,
            dtype=self.config.dtype
        )

    def add_to_model_with_buckets(self, encoder_inputs, decoder_inputs, targets, weights):
        all_inputs = encoder_inputs + decoder_inputs + targets + weights
        losses = []
        outputs = []

        # Create the internal multi-layer cell
        if self.config.num_layers == 1:
            cell = tf.contrib.rnn.BasicLSTMCell(self.config.size)
        else:
            single_cell = tf.contrib.rnn.BasicLSTMCell(self.config.size)
            cell = tf.contrib.rnn.MultiRNNCell([single_cell for _ in self.config.num_layers])

        with tf.variable_scope("en_de_model") as model_scope:
            for i, bucket in enumerate(self.config.buckets):
                buckets_outputs, _ = lambda x, y: self.add_embeddings(x, y, cell, projection, do_decode)
                outputs.append(buckets_outputs)
                losses.append(sequence_loss(
                                outputs[-1],
                                targets[:buckets[1]],
                                weights[:buckets[1]],
                                softmax_loss_function=self.add_loss_op
                                ))
        return outputs, losses

    # Add model
    def add_model(self, cell, projection):
        encoder_inputs = self.feed_dict['encoder_inputs']
        decoder_inputs = self.feed_dict['decoder_inputs']
        to_weights     = self.feed_dict['to_weights']
        targets        = self.feed_dict['targets']

        self.outputs, self.losses = self.add_to_model_with_buckets(
                encoder_inputs, decoder_inputs, targets, weights
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

    def create_model(self, is_decode):
        data_set = self.load_data()
        self.add_placeholders()

        self.feed_dict = self.create_feed_dict()

        output_projection = self.add_projection()
        self.add_model(output_projection, do_decode)

    def __init__(self, do_decode=False):
        self.config = Config
