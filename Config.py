import tensorflow as tf

class Config(object):
    
    # NMT hyperparameters
    batch_size     = 64
    max_epochs     = 1
    early_stopping = 2
    dropout        = 0.9
    lr             = 0.001
    l2             = 0.001
    
    # Encoding parameters
    encode_layers                = 3
    encode_num_steps             = 10
    encode_hidden_size           = 50
    
    # Encoding parameters
    decode_layers                = 3
    encode_num_steps             = 10
    decode_hidden_size           = 50
    
    dtype = tf.float32
    