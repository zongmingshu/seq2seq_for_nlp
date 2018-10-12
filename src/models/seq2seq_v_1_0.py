
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, LSTMStateTuple, DropoutWrapper
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense

import config


class seq2seq(object):

    def __init__(self, is_inference=False):
        self.bidirection = config.bidirection
        self.attention = config.attention
        self.hidden_size = config.hidden_size

        # h_(j): encoder output at j slide
        # s_(i-1): decoder state at i - 1 slide
        # e_(i,j) = W^T*(U^T*s_(i-1) + V^T*h_(j))
        # a_(i,j) = softmax(e_(i,j))
        # attention_num_units: dimension of U^T, V^T and W^T
        self.attention_num_units = config.attention_num_units

        # but still confused about this
        self.attention_layer_size = config.attention_layer_size

        self.layer_size = config.layer_size
        self.encoder_vocab_size = config.encoder_vocab_size
        self.decoder_vocab_size = config.decoder_vocab_size
        self.embedding_dim = config.embedding_dim
        self.grad_clip = config.grad_clip
        self.time_major = config.time_major
        self.is_inference = is_inference
        # define inputs
        self.input_x = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')
        self.start_tokens = tf.placeholder(tf.int32, shape=[None], name='start_tokens')
        self.end_token = tf.placeholder(tf.int32, shape=[], name='end_token')
        self.maximum_iterations = tf.placeholder(tf.int32, shape=[], name='maximum_iterations')
        self.target_ids = tf.placeholder(tf.int32, shape=[None, None], name='target_ids')
        self.encoder_seq_length = tf.placeholder(tf.int32, shape=[None], name='encoder_seq_length')
        self.decoder_seq_length = tf.placeholder(tf.int32, shape=[None], name='decoder_seq_length')
        self.batch_size = tf.placeholder(tf.int32, shape=[], name='batch_size')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        # define embedding layer
        with tf.variable_scope('embedding', reuse=None):
            self.encoder_embedding = tf.Variable(
                tf.truncated_normal(shape=[self.encoder_vocab_size, self.embedding_dim], stddev=0.1), 
                name='encoder_embedding')
            self.decoder_embedding = tf.Variable(
                tf.truncated_normal(shape=[self.decoder_vocab_size, self.embedding_dim], stddev=0.1),
                name='decoder_embedding')

        # define encoder
        with tf.variable_scope('encoder', reuse=None):
            encoder = self._get_simple_lstm(self.hidden_size, self.layer_size)

            with tf.device('/cpu:0'):
                input_x_embedded = tf.nn.embedding_lookup(self.encoder_embedding, self.input_x)

            '''if self.time_major:
                input_x_embedded = tf.transpose(input_x_embedded, [1, 0, 2])
            # [batch_size, 1, time, embedding_dim]
            input_x_embedded = tf.expand_dims(input_x_embedded, 1)
            input_x_embedded = tf.contrib.slim.conv2d(
                input_x_embedded,
                self.embedding_dim, [1, 3], stride=[1, 1], padding='SAME',
                scope='3-gram')
            input_x_embedded = tf.contrib.slim.batch_norm(input_x_embedded)
            input_x_embedded = tf.squeeze(input_x_embedded, [1])
            if self.time_major:
                input_x_embedded = tf.transpose(input_x_embedded, [1, 0, 2])'''

            if not self.bidirection:
                encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                    encoder, input_x_embedded, dtype=tf.float32, time_major=self.time_major)
            else:
                encoder_f_cell = LSTMCell(self.hidden_size)
                encoder_b_cell = LSTMCell(self.hidden_size)
                if not self.is_inference:
                    encoder_f_cell = DropoutWrapper(encoder_f_cell, output_keep_prob=config.keep_prob)
                    encoder_b_cell = DropoutWrapper(encoder_b_cell, output_keep_prob=config.keep_prob)
                # T * B * D, T * B * D, B * D, B * D
                (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
                    tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=encoder_f_cell,
                        cell_bw=encoder_b_cell,
                        inputs=input_x_embedded,
                        sequence_length=self.encoder_seq_length,
                        dtype=tf.float32, time_major=self.time_major)
                encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
                encoder_final_state_c = tf.concat(
                    (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
                encoder_final_state_h = tf.concat(
                    (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
                encoder_final_state = LSTMStateTuple(
                    c=encoder_final_state_c,
                    h=encoder_final_state_h
                )
                encoder_final_state = (encoder_final_state, )
            self.encoder_outputs = encoder_outputs
            self.encoder_final_state = encoder_final_state

        with tf.variable_scope('decoder', reuse=None):
            # define decoder
            # if self.is_inference:
            #     helper = GreedyEmbeddingHelper(self.decoder_embedding, self.start_tokens, self.end_token)
            # else:
            if True:
                '''with tf.device('/cpu:0'):
                    target_embeddeds = tf.nn.embedding_lookup(self.decoder_embedding, self.target_ids)
                if self.time_major:
                    target_embeddeds = tf.transpose(target_embeddeds, [1, 0, 2])
                helper = TrainingHelper(target_embeddeds, self.decoder_seq_length)'''
                sos_time_slice = tf.ones([self.batch_size], dtype=tf.int32, name='sos') * config.sos
                sos_step_embedded = tf.nn.embedding_lookup(self.decoder_embedding, sos_time_slice)
                def initial_fn():
                    initial_elements_finished = (0 >= self.decoder_seq_length)
                    if self.time_major:
                        initial_inputs = tf.concat((sos_step_embedded, self.encoder_outputs[0]), 1)
                    else:
                        initial_inputs = tf.concat((sos_step_embedded, self.encoder_outputs[:, 0, :]), 1)
                    return initial_elements_finished, initial_inputs

                def sample_fn(time, outputs, state):
                    prediction_ids = tf.to_int32(tf.argmax(outputs, axis=1))
                    # elements_finished = (time >= self.decoder_seq_length)
                    # all_finished = tf.reduce_all(elements_finished)
                    return prediction_ids

                def next_inputs_fn(time, outputs, state, sample_ids):
                    pred_embedding = tf.nn.embedding_lookup(self.decoder_embedding, sample_ids)
                    if self.time_major:
                        next_inputs = tf.concat((pred_embedding, self.encoder_outputs[time]), 1)
                    else:
                        next_inputs = tf.concat((pred_embedding, self.encoder_outputs[:, time, :]), 1)
                    elements_finished = (time >= self.decoder_seq_length)
                    next_state = state
                    return elements_finished, next_inputs, next_state

                helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)

            fc_layer = Dense(self.decoder_vocab_size)

            # in decoder, previous output as current input,
            # so decoder's num_hidden must equals encoder's num_hidden
            if self.bidirection:
                decoder_cell = self._get_simple_lstm(self.hidden_size * 2, self.layer_size)
            else:
                decoder_cell = self._get_simple_lstm(self.hidden_size, self.layer_size)

            if self.attention:
                memory = encoder_outputs
                if self.time_major:
                    # time major -> batch major
                    memory = tf.transpose(memory, [1, 0, 2])
                attention_mechanism = BahdanauAttention(
                    num_units=self.attention_num_units, memory=memory,
                    memory_sequence_length=self.encoder_seq_length)
                attention_cell = AttentionWrapper(
                    decoder_cell, attention_mechanism,
                    attention_layer_size=self.attention_layer_size)
                decoder = BasicDecoder(
                    cell=attention_cell, helper=helper,
                    initial_state=attention_cell.zero_state(
                        dtype=tf.float32, batch_size=self.batch_size
                    # ).clone(cell_state=encoder_final_state),
                    ),
                    output_layer=fc_layer)
            else:
                decoder = BasicDecoder(
                    cell=decoder_cell,
                    helper=helper,
                    initial_state=encoder_final_state,
                    output_layer=fc_layer)

            self.logits, self.final_state, self.final_sequence_lengths =\
                dynamic_decode(
                    decoder,
                    output_time_major=self.time_major,
                    maximum_iterations=self.maximum_iterations)

        self.prob = tf.nn.softmax(self.logits.rnn_output)
        self.output_id = self.logits.sample_id


    def _get_simple_lstm(self, rnn_size, layer_size):
        lstm_layers = [LSTMCell(rnn_size) for _ in xrange(layer_size)]
        return MultiRNNCell(lstm_layers)


