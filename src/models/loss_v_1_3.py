
import numpy as np
import tensorflow as tf

import config


def loss_and_train(model):
    # debug
    '''with tf.variable_scope('debug', reuse=None):
        batch_major_outputs = model.encoder_outputs
        if model.time_major:
            batch_major_outputs = tf.transpose(batch_major_outputs, [1, 0, 2])
        if model.bidirection:
            encoder_weights = tf.Variable(
                tf.truncated_normal(shape=[model.hidden_size * 2, model.decoder_vocab_size], stddev=0.1),
                name='encoder_weights')
            batch_major_outputs = tf.reshape(batch_major_outputs, [-1, model.hidden_size * 2])
        else:
            encoder_weights = tf.Variable(
                tf.truncated_normal(shape=[model.hidden_size, model.decoder_vocab_size], stddev=0.1),
                name='encoder_weights')
            batch_major_outputs = tf.reshape(batch_major_outputs, [-1, self.hidden_size])
        model.encoder_logits_flat = tf.matmul(batch_major_outputs, encoder_weights)
        model.encoder_prob = tf.nn.softmax(model.encoder_logits_flat)'''
    # debug

    with tf.variable_scope('loss_and_train', reuse=None):
        targets = tf.reshape(model.target_seq, [-1])
        seq_logits_flat = tf.reshape(model.seq_logits.rnn_output, [-1, model.decoder_vocab_size])

        # model.loss = tf.losses.sparse_softmax_cross_entropy(targets, logits_flat)
        # model.loss = tf.reduce_mean(model.loss)
        model.mask = tf.to_float(tf.not_equal(model.target_seq, 0))
        # model.mask = tf.to_float(tf.greater(model.target_seq + 1, 0))
        if model.time_major:
            model.mask = tf.transpose(model.mask, [1, 0])
            model.seq_loss = tf.contrib.seq2seq.sequence_loss(
                tf.transpose(model.seq_logits.rnn_output, [1, 0, 2]),
                tf.transpose(model.target_seq, [1, 0]),
                weights=model.mask)
        else:
            model.seq_loss = tf.contrib.seq2seq.sequence_loss(
                model.seq_logits.rnn_output,
                model.target_seq,
                weights=model.mask)

        model.label_class_loss = tf.losses.sparse_softmax_cross_entropy(model.target_label, model.predict_logits)

        model.loss = model.seq_loss + config.label_class_loss_factor * model.label_class_loss

        # debug

        # crf
        '''lstm_logits = model.logits.rnn_output
        lstm_targets = model.target_seq
        if model.time_major:
            lstm_logits = tf.transpose(lstm_logits, [1, 0, 2])
            lstm_targets = tf.transpose(lstm_targets, [1, 0])
        model.log_likelihood, model.trans_params = tf.contrib.crf.crf_log_likelihood(
            lstm_logits, lstm_targets, model.encoder_seq_length)
        model.loss = tf.reduce_mean(-model.log_likelihood)
        model.crf_prob, _ = tf.contrib.crf.crf_decode(lstm_logits, model.trans_params, model.encoder_seq_length)
        if model.time_major:
            model.crf_prob = tf.transpose(model.crf_prob, [1, 0])'''

        # ce
        # model.loss = tf.losses.sparse_softmax_cross_entropy(targets, model.encoder_logits_flat)
        # model.loss = tf.reduce_mean(model.loss)

        # debug

        # define train op
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(model.loss, tvars), model.grad_clip)
        optimizer = tf.train.AdamOptimizer(model.learning_rate)
        # gradients_and_variables = optimizer.compute_gradients(model.loss, tf.trainable_variables())
        model.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=model.global_step)
        # model.train_op = optimizer.apply_gradients(gradients_and_variables, global_step=model.global_step)
        return model.train_op


