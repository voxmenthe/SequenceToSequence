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
    def add_embedding(self, en_inputs, de_inputs, cell, projection, do_decode):
        return tf.contrib.legacy_seq2seq.embedding_attention(
            en_inputs,
            de_inputs,
            cell,
            num_encoder_symbols=self.config.from_vocab_size,
            num_decoder_symbols=self.config.to_vocab_size,
            embedding_size=self.config.size,
            output_projection=projection
            feed_previous=do_decode,
            dtype=self.config.dtype
        )

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
    def add_model(self, do_decode):
        encoder_inputs = self.feed_dict['encoder_inputs']
        decoder_inputs = self.feed_dict['decoder_inputs']
        to_weights     = self.feed_dict['to_weights']
        targets        = self.feed_dict['targets']

        self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
            encoder_inputs, decoder_inputs, targets, to_weights,
            self.config.buckets, lambda x, y: add_embedding(x, y, do_decode),
            softmax_loss_function=add_loss_op
        )

        if do_decode:
            params = tf.trainable_variables()
            self.gradient_norms = []
            self.updates = []

            self.lr = tf.Variable(float(self.config.lr), trainable=False, dtype=self.config.dtype)
            self.global_step = tf.Variable(0, trainable=False)

            optimizer = tf.train.GradientDescentOptimizer(self.config.lr)

            for b in xrange(len(self.config.buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                self.config.max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(optimier.apply_gradients(
                    zip(clipped_gradients, params),  global_step=self.global_step
                ))

    def __init__(self):
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
