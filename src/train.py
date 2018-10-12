
import numpy as np
import tensorflow as tf

import importlib
import sys
import os

curPath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(curPath, './'))
sys.path.append(os.path.join(curPath, './models'))
sys.path.append(os.path.join(curPath, './tools'))

from datetime import datetime

import helpers
import preprocess
import config
import module_define as md


def schedule_lr(epoch):
    if epoch < 5:
        return 2e-3
    elif epoch < 10:
        return 1e-3
    elif epoch < 20:
        return 2e-4
    else:
        return 1e-4
    # return 2e-3


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5'

    sentences, seq_labels, vocabulary, rev_vocabulary = preprocess.build_sentences_set(sos_eos=False)
    print('Train data num: %d' % len(sentences))
    max_seq_len = config.max_seq_len
    sentences, sentences_len = helpers.batch(sentences, max_sequence_length=max_seq_len)
    seq_labels, seq_labels_len = helpers.batch(seq_labels, max_sequence_length=max_seq_len)
    labels = np.array(range(0, sentences.shape[1]))

    if not os.path.isdir(md.ckpt_path):
        os.makedirs(md.ckpt_path)
    if not os.path.isdir(md.log_path):
        os.makedirs(md.log_path)

    with tf.Graph().as_default() as graph:
        seq2seq_module = importlib.import_module(md.model)
        loss_module = importlib.import_module(md.loss)
        model = seq2seq_module.seq2seq()
        train_op = loss_module.loss_and_train(model)

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=4)
        summary_writer = tf.summary.FileWriter(md.log_path, graph)
        # tf.summary.scalar('loss', model.loss)
        for i in range(0, config.num_gpu):
            tf.summary.scalar('loss_%d' % i, model.loss_splits[i])
        summary_op = tf.summary.merge_all()

        gpu_config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=gpu_config) as session:
            session.run(tf.global_variables_initializer())
            session.run(tf.local_variables_initializer())

            metagraph_saved = False
            for epoch in range(0, config.num_epoch):
                for batch in range(0, config.epoch_size):
                    step = session.run(model.global_step, feed_dict=None)

                    indices = np.random.randint(sentences.shape[1], size=config.batch_size)
                    batch_seq = sentences[:, indices]
                    batch_seq_labels = seq_labels[:, indices]
                    batch_labels = labels[indices]

                    batch_seq = batch_seq if config.time_major else batch_seq.T
                    batch_seq_labels = batch_seq_labels if config.time_major else batch_seq_labels.T
                    batch_seq_targets = batch_seq_labels if not config.auto_encoder else batch_seq

                    '''_, loss, seq_prob, encoder_outputs, encoder_final_state, summary_str = session.run([
                        train_op, model.loss, model.seq_prob,
                        model.encoder_outputs, model.encoder_final_state,
                        summary_op],
                        feed_dict={
                            model.input_seq: batch_seq,
                            model.target_seq: batch_seq_targets,
                            model.target_label: batch_labels,
                            model.encoder_seq_length: max_seq_len * np.ones(config.batch_size),
                            model.decoder_seq_length: max_seq_len * np.ones(config.batch_size),
                            model.batch_size: config.batch_size,
                            model.maximum_iterations: max_seq_len,
                            model.learning_rate: schedule_lr(epoch)})'''
                    _, loss_splits, seq_prob_splits, encoder_outputs_splits, encoder_final_state_splits, summary_str = session.run([
                        train_op, model.loss_splits, model.seq_prob_splits,
                        model.encoder_outputs_splits, model.encoder_final_state_splits,
                        summary_op],
                        feed_dict={
                            model.input_seq: batch_seq,
                            model.target_seq: batch_seq_targets,
                            model.target_label: batch_labels,
                            model.encoder_seq_length: max_seq_len * np.ones(config.batch_size),
                            model.decoder_seq_length: max_seq_len * np.ones(config.batch_size),
                            model.batch_size: config.batch_size,
                            model.maximum_iterations: max_seq_len,
                            model.learning_rate: schedule_lr(epoch)})
                    loss = np.asarray(loss_splits)
                    loss = np.mean(loss)
                    print('Epoch: [%d/%d]; Batch: [%d/%d]; Loss: %.4f' % (
                        epoch, config.num_epoch, batch, config.epoch_size, loss))
                    summary_writer.add_summary(summary_str, global_step=step)

                print('Saving model')
                saver.save(
                    session,
                    os.path.join(md.ckpt_path, '%s.ckpt' % md.model_name),
                    global_step=step, write_meta_graph=False)
                if not metagraph_saved:
                    print('Saving metagraph')
                    saver.export_meta_graph(os.path.join(md.ckpt_path, '%s.meta' % md.model_name))
                    metagraph_saved = True


