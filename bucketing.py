# Buckets End To End


# A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
# to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
# the size if i-th training bucket, as used later.
train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                       for i in xrange(len(train_bucket_sizes))]