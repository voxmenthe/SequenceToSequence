import tensorflow as tf

class Config(object):

    # Source and Target files
    from_train_file='data/dev.en'
    to_train_file='data/dev.vi'

    # Special characters and ID's
    _PAD = b"_PAD"
    _GO = b"_GO"
    _EOS = b"_EOS"
    _UNK = b"_UNK"
    _START_VOCAB = [_PAD, _GO, _EOS, _UNK]

    PAD_ID = 0
    GO_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    # NMT hyperparameters
    batch_size                   = 64
    max_epochs                   = 1
    early_stopping               = 2
    dropout                      = 0.9
    lr                           = 0.5
    l2                           = 0.001
    learning_rate_decay          = 0.99
    batch_size                   = 32
    size                         = 1024
    num_layers                   = 3
    from_vocab_size              = 10000
    to_vocab_size                = 10000
    data_dir                     = "data/"
    dev_dir                      = "data/"
    max_train_data_size          = 200
    steps_per_checkpoint         = 5
    forward_only                 = True

    # Buckets
    buckets = [(5,10), (10,15)]

    # Other config variables
    num_samples                  = 512

    # Encoding parameters
    encode_layers                = 3
    encode_num_steps             = 10
    encode_hidden_size           = 50

    # Encoding parameters
    decode_layers                = 3
    encode_num_steps             = 10
    decode_hidden_size           = 50

    dtype = tf.float32
