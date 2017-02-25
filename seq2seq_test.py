import tensorflow as tf
import os
import math
import random
import time
from seq2seq import Seq2SeqModel
from DataUtils import DataUtils
from Config import Config

self.config = Config()

def create_model():
    model = Seq2SeqModel(is_decode=False)
    return model

def train():
    with tf.session() as sess:
        model = create_model()

        dev_set = read_data('data/dev.en', 'data/dev.vi')
        train_set = read_data('data/tst2012.en', 'data/tst2012.vi', self.config.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(self.config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]
        current_step = 0
        while current_step < 50:
            random_number_01 = np.random.random_sample()

            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                           if train_buckets_scale[i] > random_number_01])

            encoder_inputs, decoder_inputs, weights = Seq2SeqModel.get_batch(train_set, bucket_id)

            _, step_loss = model.step(sess, encoder_inputs, decoder_inputs, weights, b)
            loss += step_loss/self.config.steps_per_checkpoint
            current_step += 1

            if current_step % self.config.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                       "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                 step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                  sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(self.config.train_dir, "translate.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                for bucket_id in xrange(len(_buckets)):
                  if len(dev_set[bucket_id]) == 0:
                    print("  eval: empty bucket %d" % (bucket_id))
                    continue
                  encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                      dev_set, bucket_id)
                  _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                               target_weights, bucket_id, True)
                  eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
                      "inf")
                  print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()

if __name__ == "__main__":
    train()
