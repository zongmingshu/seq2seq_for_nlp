
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
        self.label_class_size = config.label_class_size

        # h_(j): encoder output at j slide
        # s_(i-1): decoder state at i-1 slide
        # e_(i,j) = W^T*(U^T*s_(i-1)+V^T*h_(j))
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
        self.input_seq = tf.placeholder(tf.int32, shape=[None, None], name='input_seq')
        self.start_tokens = tf.placeholder(tf.int32, shape=[None], name='start_tokens')
        self.end_token = tf.placeholder(tf.int32, shape=[], name='end_token')
        self.maximum_iterations = tf.placeholder(tf.int32, shape=[], name='maximum_iterations')
        self.target_seq = tf.placeholder(tf.int32, shape=[None, None], name='target_seq')
        self.target_label = tf.placeholder(tf.int32, shape=[None], name='target_label')
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
            if config.auto_encoder:
                self.decoder_embedding = self.encoder_embedding
            else:
                self.decoder_embedding = tf.Variable(
                    tf.truncated_normal(shape=[self.decoder_vocab_size, self.embedding_dim], stddev=0.1),
                    name='decoder_embedding')

        # define encoder
        with tf.variable_scope('encoder', reuse=None):
            encoder = self._get_simple_lstm(self.hidden_size, self.layer_size)

            with tf.device('/cpu:0'):
                input_seq_embedded = tf.nn.embedding_lookup(self.encoder_embedding, self.input_seq)

            # 3-gram emulation
            if self.time_major:
                input_seq_embedded = tf.transpose(input_seq_embedded, [1, 0, 2])
            # [batch_size, 1, time, embedding_dim]
            input_seq_embedded = tf.expand_dims(input_seq_embedded, 1)
            input_seq_embedded = tf.contrib.slim.conv2d(
                input_seq_embedded,
                self.embedding_dim, [1, 3], stride=[1, 1], padding='SAME',
                scope='3-gram')
            input_seq_embedded = tf.contrib.slim.batch_norm(input_seq_embedded)
            input_seq_embedded = tf.squeeze(input_seq_embedded, [1])
            if self.time_major:
                input_seq_embedded = tf.transpose(input_seq_embedded, [1, 0, 2])

            if not self.bidirection:
                encoder_cell = self._get_simple_lstm(self.hidden_size, self.layer_size)
                encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                    encoder_cell, input_seq_embedded, dtype=tf.float32, time_major=self.time_major)
                encoder_final_state_c = tf.concat([fs.c for fs in encoder_final_state], axis=1)
                encoder_final_state_h = tf.concat([fs.h for fs in encoder_final_state], axis=1)
                c_fc = tf.contrib.slim.fully_connected(encoder_final_state_c, self.hidden_size)
                h_fc = tf.contrib.slim.fully_connected(encoder_final_state_h, self.hidden_size)
            else:
                # encoder_f_cell = LSTMCell(self.hidden_size, reuse=reuse)
                # encoder_b_cell = LSTMCell(self.hidden_size, reuse=reuse)
                encoder_f_cell = self._get_simple_lstm(self.hidden_size, self.layer_size)
                encoder_b_cell = self._get_simple_lstm(self.hidden_size, self.layer_size)
                if not self.is_inference:
                    encoder_f_cell = DropoutWrapper(encoder_f_cell, output_keep_prob=config.keep_prob)
                    encoder_b_cell = DropoutWrapper(encoder_b_cell, output_keep_prob=config.keep_prob)
                # T * B * D, T * B * D, B * D, B * D
                (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
                    tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=encoder_f_cell,
                        cell_bw=encoder_b_cell,
                        inputs=input_seq_embedded,
                        sequence_length=self.encoder_seq_length,
                        dtype=tf.float32, time_major=self.time_major)
                encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

                encoder_final_state_c = tf.concat([fs.c for fs in encoder_fw_final_state] + [bs.c for bs in encoder_bw_final_state], axis=1)
                encoder_final_state_h = tf.concat([fs.h for fs in encoder_fw_final_state] + [bs.h for bs in encoder_bw_final_state], axis=1)
                c_fc = tf.contrib.slim.fully_connected(encoder_final_state_c, self.hidden_size * 2)
                h_fc = tf.contrib.slim.fully_connected(encoder_final_state_h, self.hidden_size * 2)

                # encoder_final_state_c = tf.concat(
                #     (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
                # encoder_final_state_c = tf.nn.l2_normalize(encoder_final_state_c, dim=1, epsilon=1e-10, name='c_norm')
                # encoder_final_state_h = tf.concat(
                #     (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
                # encoder_final_state_h = tf.nn.l2_normalize(encoder_final_state_h, dim=1, epsilon=1e-10, name='h_norm')
                # encoder_final_state = LSTMStateTuple(
                    # c=encoder_final_state_c,
                    # h=encoder_final_state_h
                # )
            encoder_final_state = LSTMStateTuple(
                c=c_fc,
                h=h_fc
            )
            # encoder_final_state = (encoder_final_state, )

            self.encoder_outputs = encoder_outputs
            self.encoder_final_state = encoder_final_state

            # label class
            if not self.bidirection:
                label_class_w = tf.Variable(
                    tf.truncated_normal(shape=[self.hidden_size, self.label_class_size], stddev=0.1),
                    name='label_class_w')
            else:
                label_class_w = tf.Variable(
                    tf.truncated_normal(shape=[self.hidden_size * 2, self.label_class_size], stddev=0.1),
                    name='label_class_w')
            label_class_b = tf.Variable(
                tf.truncated_normal(shape=[self.label_class_size], stddev=0.1),
                name='label_class_b')
            # h = encoder_final_state[0].h
            h = encoder_final_state.h
            predict_logits = tf.matmul(h, label_class_w)
            predict_logits = tf.nn.bias_add(predict_logits, label_class_b)
            predict_logits = tf.nn.relu(predict_logits)
            self.predict_logits = predict_logits

        with tf.variable_scope('decoder', reuse=None):
            if True:
                sos_time_slice = tf.ones([self.batch_size], dtype=tf.int32, name='sos') * config.sos
                sos_step_embedded = tf.nn.embedding_lookup(self.decoder_embedding, sos_time_slice)
                def initial_fn():
                    initial_elements_finished = (0 >= self.decoder_seq_length)
                    # initial_inputs = tf.concat((sos_step_embedded, encoder_final_state[0].h), 1)
                    initial_inputs = tf.concat((sos_step_embedded, encoder_final_state.h), 1)
                    return initial_elements_finished, initial_inputs

                def sample_fn(time, outputs, state):
                    prediction_ids = tf.to_int32(tf.argmax(tf.nn.softmax(outputs), axis=1))
                    return prediction_ids

                def next_inputs_fn(time, outputs, state, sample_ids):
                    pred_embedding = tf.nn.embedding_lookup(self.decoder_embedding, sample_ids)
                    # next_inputs = tf.concat((pred_embedding, encoder_final_state[0].h), 1)
                    next_inputs = tf.concat((pred_embedding, encoder_final_state.h), 1)
                    elements_finished = (time >= self.decoder_seq_length)
                    next_state = state
                    return elements_finished, next_inputs, next_state

                helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)

            fc_layer = Dense(self.decoder_vocab_size)

            # in decoder, previous output as current input,
            # so decoder's num_hidden must equals encoder's num_hidden
            if self.bidirection:
                # decoder_cell = self._get_simple_lstm(self.hidden_size * 2, self.layer_size, reuse)
                decoder_cell = LSTMCell(self.hidden_size * 2)
            else:
                # decoder_cell = self._get_simple_lstm(self.hidden_size, self.layer_size, reuse)
                decoder_cell = LSTMCell(self.hidden_size)

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
                    ).clone(cell_state=encoder_final_state),
                    # ),
                    output_layer=fc_layer)
            else:
                decoder = BasicDecoder(
                    cell=decoder_cell,
                    helper=helper,
                    initial_state=encoder_final_state,
                    output_layer=fc_layer)

            self.seq_logits, self.final_state, self.final_sequence_lengths =\
                dynamic_decode(
                    decoder,
                    output_time_major=self.time_major,
                    maximum_iterations=self.maximum_iterations)

        self.seq_prob = tf.nn.softmax(self.seq_logits.rnn_output)
        self.seq_output_id = self.seq_logits.sample_id


    def _get_simple_lstm(self, rnn_size, layer_size):
        lstm_layers = [LSTMCell(rnn_size) for _ in range(layer_size)]
        return MultiRNNCell(lstm_layers)


