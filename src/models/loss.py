
import numpy as np
import tensorflow as tf

import config


def loss_and_train(model):
    gradient_splits = []
    model.seq_loss_splits = []
    model.label_class_loss_splits = []
    model.loss_splits = []
    reuse_variables = False
    optimizer = tf.train.AdamOptimizer(model.learning_rate)
    for i in range(0, config.num_gpu):
        with tf.device('/gpu:%d' % i):
            _loss(
                model, optimizer, gradient_splits,
                model.seq_logits_splits[i],
                model.predict_logits_splits[i],
                model.target_seq_splits[i],
                model.target_label_splits[i],
                reuse_variables)
            reuse_variables = True

    average_g = []
    for grad_and_var in zip(*gradient_splits):
        grads = []
        for g, var in grad_and_var:
            grads.append(tf.expand_dims(g, 0))
        grads = tf.concat(axis=0, values=grads)
        grads = tf.reduce_mean(grads, 0)

        v = grad_and_var[0][1]
        grad_and_var = (grads, v)
        average_g.append(grad_and_var)

    # define train op
    # tvars = tf.trainable_variables()
    # grads, _ = tf.clip_by_global_norm(tf.gradients(model.loss, tvars), model.grad_clip)
    # gradients_and_variables = optimizer.compute_gradients(model.loss, tf.trainable_variables())
    # model.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=model.global_step)
    clip_g, _ = tf.clip_by_global_norm([g for g, v in average_g], model.grad_clip)
    model.train_op = optimizer.apply_gradients(zip(clip_g, tf.trainable_variables()), global_step=model.global_step)
    return model.train_op


def _loss(
    model, optimizer, gradient_splits,
    seq_logits_split,
    predict_logits_split,
    target_seq_split,
    target_label_split,
    reuse):
    with tf.variable_scope('loss_and_train', reuse=reuse):
        targets = tf.reshape(target_seq_split, [-1])
        seq_logits_flat = tf.reshape(
            seq_logits_split.rnn_output,
            [-1, model.decoder_vocab_size])

        mask = tf.to_float(tf.not_equal(target_seq_split, 0))
        if model.time_major:
            mask = tf.transpose(mask, [1, 0])
            seq_loss = tf.contrib.seq2seq.sequence_loss(
                tf.transpose(seq_logits_split.rnn_output, [1, 0, 2]),
                tf.transpose(target_seq_split, [1, 0]),
                weights=mask)
        else:
            seq_loss = tf.contrib.seq2seq.sequence_loss(
                seq_logits_split.rnn_output,
                target_seq_split,
                weights=mask)

        label_class_loss = tf.losses.sparse_softmax_cross_entropy(
            target_label_split,
            predict_logits_split)

        model.seq_loss_splits.append(seq_loss)
        model.label_class_loss_splits.append(label_class_loss)

        loss = seq_loss + config.label_class_loss_factor * label_class_loss
        model.loss_splits.append(loss)
        # model.loss = model.label_class_loss

        gradient = optimizer.compute_gradients(loss, tf.trainable_variables())
        gradient_splits.append(gradient)


