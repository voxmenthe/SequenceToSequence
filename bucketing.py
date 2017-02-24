# Buckets End To End






# Read data into buckets and compute their sizes.
print ("Reading development and training data (limit: %d)."
       % FLAGS.max_train_data_size)
dev_set = read_data(from_dev, to_dev)
train_set = read_data(from_train, to_train, FLAGS.max_train_data_size)
train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
train_total_size = float(sum(train_bucket_sizes))

# A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
# to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
# the size if i-th training bucket, as used later.
train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                       for i in xrange(len(train_bucket_sizes))]


# In the training loop:
# Choose a bucket according to data distribution. We pick a random number
# in [0, 1] and use the corresponding interval in train_buckets_scale.
random_number_01 = np.random.random_sample()
bucket_id = min([i for i in xrange(len(train_buckets_scale))
               if train_buckets_scale[i] > random_number_01])

