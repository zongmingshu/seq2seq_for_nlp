
import numpy as np
import scipy
import tensorflow as tf

import importlib
import sys
import os

curPath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(curPath, './'))
sys.path.append(os.path.join(curPath, './models'))
sys.path.append(os.path.join(curPath, './tools'))

from datetime import datetime
from itertools import chain

import preprocess
import helpers
import config
import module_define as md


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3, 4'

    _sentences, _seq_labels, vocabulary, rev_vocabulary = preprocess.build_test_set(sos_eos=False)
    print('Train data num: %d' % len(_sentences))
    sentences = _sentences
    seq_labels = _seq_labels
    max_seq_len = config.max_seq_len

    sentences, sentences_len = helpers.batch(sentences, max_sequence_length=max_seq_len)
    seq_labels, seq_labels_len = helpers.batch(seq_labels, max_sequence_length=max_seq_len)

    with tf.Graph().as_default():
        seq2seq_module = importlib.import_module(md.model)
        loss_module = importlib.import_module(md.loss)
        model = seq2seq_module.seq2seq(is_inference=True)
        train_op = loss_module.loss_and_train(model)

        # saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=4)
        saver = tf.train.Saver(tf.trainable_variables())

        gpu_config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=gpu_config) as session:
            session.run(tf.global_variables_initializer())
            session.run(tf.local_variables_initializer())

            print(tf.train.latest_checkpoint(md.ckpt_path))
            saver.restore(session, tf.train.latest_checkpoint(md.ckpt_path))

            batch_size = 64
            indices = np.random.randint(sentences.shape[1], size=batch_size)
            batch_seq = sentences[:, indices]
            batch_seq_labels = seq_labels[:, indices]
            batch_seq_len = np.asarray(sentences_len)[indices]

            batch_seq = batch_seq if config.time_major else batch_seq.T
            batch_seq_labels = batch_seq_labels if config.time_major else batch_seq_labels.T

            '''seq_prob, encoder_outputs, encoder_final_state = session.run(
                [model.seq_prob, model.encoder_outputs, model.encoder_final_state],
                feed_dict={
                    model.input_seq: batch_seq,
                    model.encoder_seq_length: max_seq_len * np.ones(batch_size),
                    model.decoder_seq_length: max_seq_len * np.ones(batch_size),
                    # model.start_tokens: config.sos * np.ones(batch_size),
                    # model.end_token: config.eos,
                    model.batch_size: batch_size,
                    model.maximum_iterations: max_seq_len})'''
            seq_prob_splits, encoder_outputs_splits, encoder_final_state_splits = session.run(
                [model.seq_prob_splits, model.encoder_outputs_splits, model.encoder_final_state_splits],
                feed_dict={
                    model.input_seq: batch_seq,
                    model.encoder_seq_length: max_seq_len * np.ones(batch_size),
                    model.decoder_seq_length: max_seq_len * np.ones(batch_size),
                    # model.start_tokens: config.sos * np.ones(batch_size),
                    # model.end_token: config.eos,
                    model.batch_size: batch_size,
                    model.maximum_iterations: max_seq_len})

            '''print(encoder_final_state)
            print('encoder_final_state.c(shape: %s):' % str(encoder_final_state[0].c.shape))
            print(encoder_final_state[0].c)
            print('encoder_final_state.h(shape: %s):' % str(encoder_final_state[0].h.shape))
            print(encoder_final_state[0].h)
            c = encoder_final_state[0].c
            h = encoder_final_state[0].h'''
            for i in range(0, config.num_gpu):
                print(encoder_final_state_splits[i][0].c.shape)
                print(encoder_final_state_splits[i][0].h.shape)
            concat_axis = (1 if config.time_major else 0)
            seq_prob = np.concatenate(seq_prob_splits, axis=concat_axis)
            encoder_outputs = np.concatenate(encoder_outputs_splits, axis=concat_axis)
            c = np.concatenate([s[0].c for s in encoder_final_state_splits])
            h = np.concatenate([s[0].h for s in encoder_final_state_splits])

            final = np.argmax(seq_prob, axis=2)
            # final = prob
            final = final if config.time_major else final.T

            total_err = 0
            for i in range(0, batch_size):
                seq_len = batch_seq_len[i]
                print('Sentence:')
                print(''.join([rev_vocabulary[s] for s in batch_seq[0: seq_len, i]]))
                if not config.auto_encoder:
                    print('Gold:')
                    print(batch_seq_labels[0: seq_len, i])
                print('Predict labels:')
                if not config.auto_encoder:
                    print(final[0: seq_len, i])
                else:
                    seq_str = [rev_vocabulary[s] for s in final[0: seq_len, i]]
                    print(''.join(seq_str))
                err = batch_seq_labels[0: seq_len, i] - final[0: seq_len, i]
                # err = batch_seq[0: seq_len, i] - final[0: seq_len, i]
                # print(err)
                err[err != 0] = 1
                total_err += np.sum(err)
                # print('Sentence label error: %d' % np.sum(err))
            print('Total error rate: %.4f' % (float(total_err) / float(np.sum(batch_seq_len))))

            dist = scipy.spatial.distance.pdist(h, metric='cosine')
            index = [[[i, j] for j in range(i + 1, batch_size)] for i in range(0, batch_size - 1)]
            index = list(chain(*index))
            index = np.asarray(index)
            sort_dist = np.argsort(dist)
            min_pairs = index[sort_dist[0: 10]]
            max_pairs = index[sort_dist[-10: -1]]

            for min_pair in min_pairs:
                print('min_pair(%f):' % dist[sort_dist[0]])
                min_seq_0 = batch_seq[0: batch_seq_len[min_pair[0]], min_pair[0]]
                min_seq_1 = batch_seq[0: batch_seq_len[min_pair[1]], min_pair[1]]
                min_str_0 = [rev_vocabulary[s] for s in min_seq_0]
                min_str_1 = [rev_vocabulary[s] for s in min_seq_1]
                print(''.join(min_str_0))
                print(''.join(min_str_1))
            for max_pair in max_pairs:
                print('max_pair(%f):' % dist[sort_dist[-1]])
                max_seq_0 = batch_seq[0: batch_seq_len[max_pair[0]], max_pair[0]]
                max_seq_1 = batch_seq[0: batch_seq_len[max_pair[1]], max_pair[1]]
                max_str_0 = [rev_vocabulary[s] for s in max_seq_0]
                max_str_1 = [rev_vocabulary[s] for s in max_seq_1]
                print(''.join(max_str_0))
                print(''.join(max_str_1))


