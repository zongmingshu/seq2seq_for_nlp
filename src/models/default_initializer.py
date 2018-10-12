
import tensorflow as tf

import config


def _embedding_initialize(model):
    model.encoder_embedding = tf.Variable(
        tf.truncated_normal(shape=[model.encoder_vocab_size, model.embedding_dim], stddev=0.1),
        name='encoder_embedding')
    if config.auto_encoder:
        model.decoder_embedding = model.encoder_embedding
    else:
        model.decoder_embedding = tf.Variable(
            tf.truncated_normal(shape=[model.decoder_vocab_size, model.embedding_dim], stddev=0.1),
            name='decoder_embedding')


