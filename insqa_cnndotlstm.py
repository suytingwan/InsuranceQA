import tensorflow as tf
import numpy
import numpy as np

class InsBiDotLSTM(object):
    def __init__(
        self, vectors, sequence_length, batch_size,
        vocab_size, embedding_size,
        filter_sizes, num_filters, l2_reg_lambda=0.0):

        self.input_x_1 = tf.placeholder(tf.int32, [batch_size, sequence_length],name='input_x_1')
        self.input_x_2 = tf.placeholder(tf.int32, [batch_size, sequence_length],name="input_x_2")
        self.input_x_3 = tf.placeholder(tf.int32, [batch_size, sequence_length],name="input_x_3")

        #self.hidden_size = 100
        self.hidden_size = 141
        self.num_layers = 1
        self.num_steps = 201

        self.conv_size = 200

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            #W = tf.Variable(
            #    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
            #    name="W")
            W = tf.constant(vectors, tf.float32, name='W')

            chars_1 = tf.nn.embedding_lookup(W, self.input_x_1)
            chars_2 = tf.nn.embedding_lookup(W, self.input_x_2)
            chars_3 = tf.nn.embedding_lookup(W, self.input_x_3)

            self.expand_chars1 = tf.expand_dims(chars_1, -1)
            self.expand_chars2 = tf.expand_dims(chars_2, -1)
            self.expand_chars3 = tf.expand_dims(chars_3, -1)

        conv_output_1 = []
        conv_output_2 = []
        conv_output_3 = []
        for i, filter_size in enumerate(filter_sizes):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W_2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),name='W2')
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]),name='b')

            conv = tf.nn.conv2d(
                       self.expand_chars1,
                       W_2,
                       strides=[1,1,1,1],
                       padding='VALID',
                       name='conv-1')
            h = tf.nn.relu(tf.nn.bias_add(conv,b), name='relu-1')
            zeros_1 = tf.zeros([batch_size,filter_size,1,num_filters])
            cat_1 = tf.concat(1,[conv, zeros_1])
            conv_output_1.append(cat_1)

            conv = tf.nn.conv2d(
                      self.expand_chars2,
                      W_2,
                      strides=[1,1,1,1],
                      padding='VALID',
                      name='conv-2')
            h = tf.nn.relu(tf.nn.bias_add(conv,b), name='relu-2')
            zeros_2 = tf.zeros([batch_size,filter_size,1,num_filters])
            cat_2 = tf.concat(1, [conv, zeros_2])
            conv_output_2.append(cat_2)

            conv = tf.nn.conv2d(
                       self.expand_chars3,
                       W_2,
                       strides=[1,1,1,1],
                       padding='VALID',
                       name='conv-3')
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu-3')
            zeros_3 = tf.zeros([batch_size,filter_size,1,num_filters])
            cat_3 = tf.concat(1, [conv,zeros_3])
            conv_output_3.append(cat_3)
        num_filters_total = num_filters * len(filter_sizes)
        conv_re_1 = tf.reshape(tf.concat(3, conv_output_1),[-1,num_filters_total])
        conv_re_2 = tf.reshape(tf.concat(3, conv_output_2),[-1,num_filters_total])
        conv_re_3 = tf.reshape(tf.concat(3, conv_output_3),[-1,num_filters_total])

        with tf.variable_scope('Medium') as scope:
            W_3 = tf.get_variable('W-3', shape=[num_filters_total, self.conv_size], initializer=tf.contrib.layers.xavier_initializer())
            b_3 = tf.Variable(tf.constant(0.1, shape=[self.conv_size]), name='b2')
            tanh_output_1 = tf.nn.tanh(tf.nn.xw_plus_b(conv_re_1, W_3, b_3))
            tanh_output_2 = tf.nn.tanh(tf.nn.xw_plus_b(conv_re_2, W_3, b_3))
            tanh_output_3 = tf.nn.tanh(tf.nn.xw_plus_b(conv_re_3, W_3, b_3))

        reshape_output_1 = tf.reshape(tanh_output_1, [-1,self.num_steps,self.conv_size])
        reshape_output_2 = tf.reshape(tanh_output_2, [-1,self.num_steps,self.conv_size])
        reshape_output_3 = tf.reshape(tanh_output_3, [-1,self.num_steps,self.conv_size])

        chars_1_ = tf.transpose(reshape_output_1,perm=[1,0,2])
        chars_2_ = tf.transpose(reshape_output_2,perm=[1,0,2])
        chars_3_ = tf.transpose(reshape_output_3,perm=[1,0,2])

        chars_11 = tf.reshape(chars_1_, shape=[-1, self.conv_size])
        chars_22 = tf.reshape(chars_2_, shape=[-1, self.conv_size])
        chars_33 = tf.reshape(chars_3_, shape=[-1, self.conv_size])

        chars1 = tf.split(split_dim=0, num_split=self.num_steps, value=chars_11)
        chars2 = tf.split(split_dim=0, num_split=self.num_steps, value=chars_22)
        chars3 = tf.split(split_dim=0, num_split=self.num_steps, value=chars_33)

        output_1 = []
        output_2 = []
        output_3 = []
        att_1 = []
        att_2 = []
        
        with tf.variable_scope('forward'):
            self.lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0)
        with tf.variable_scope('backward'):
            self.lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0)

        lstm_state_fw = self.lstm_fw_cell.zero_state(batch_size,tf.float32)
        lstm_state_bw = self.lstm_bw_cell.zero_state(batch_size,tf.float32)
        input_length = tf.fill([batch_size,],self.num_steps)

        outputs_1, _, _ = tf.nn.bidirectional_rnn(self.lstm_fw_cell,
                                                      self.lstm_bw_cell,
                                                      chars1,
                                                      sequence_length=input_length,
                                                      initial_state_fw=lstm_state_fw,
                                                      initial_state_bw=lstm_state_bw,
                                                      dtype=tf.float32)

        tf.get_variable_scope().reuse_variables()

        outputs_2,_, _ = tf.nn.bidirectional_rnn(self.lstm_fw_cell,
                                                     self.lstm_bw_cell,
                                                     chars2,
                                                     sequence_length=input_length,
                                                     initial_state_fw=lstm_state_fw,
                                                     initial_state_bw=lstm_state_bw,
                                                     dtype=tf.float32)
        tf.get_variable_scope().reuse_variables()

        outputs_3, _, _ = tf.nn.bidirectional_rnn(self.lstm_fw_cell,
                                                      self.lstm_bw_cell,
                                                      chars3,
                                                      sequence_length=input_length,
                                                      initial_state_fw=lstm_state_fw,
                                                      initial_state_bw=lstm_state_bw,
                                                      dtype=tf.float32)

        Wa = tf.Variable(
                tf.random_uniform([self.hidden_size*2,10], -1.0, 1.0),
                name='Wa')
        Wq = tf.Variable(
                tf.random_uniform([self.hidden_size*2,10], -1.0, 1.0),
                name='Wq')
        wm = tf.Variable(
                tf.random_uniform([10,1],-1.0,1.0),
                name='wm')
            
        trans_out_1 = tf.transpose(outputs_1, perm=[1,0,2])
        last_output_1 = outputs_1[-1]

        outputs_1 = tf.reshape(trans_out_1,[-1,self.num_steps,self.hidden_size*2,1])          

        pooled_1 = tf.nn.max_pool(
                outputs_1,
                ksize=[1, self.num_steps, 1, 1],
                strides=[1,1,1,1],
                padding='VALID',
                name='pool-1'
                )
        pool_flat_1 = tf.reshape(pooled_1, [-1,self.hidden_size*2])
        #print pooled_1.get_shape().as_list()
            
            
        trans_out_2 = tf.transpose(outputs_2, perm=[1,0,2])
        for time_step in range(self.num_steps):
            cell_out = tf.reshape(outputs_2[time_step],[-1,self.hidden_size*2])
            att = tf.reduce_sum(cell_out * last_output_1, 1)
            #print att.get_shape().as_list()
            att = tf.reshape(att,[-1,1])
            att_1.append(att)

        s_att_1 = tf.concat(1, att_1)
        s_att_1 = tf.reshape(s_att_1, [-1,self.num_steps,1])
        output_att_2 = trans_out_2 * s_att_1
        pool_reshape_2 = tf.reshape(output_att_2, [-1, self.num_steps, self.hidden_size*2, 1])
        
        pooled_2 = tf.nn.max_pool(
                pool_reshape_2,
                ksize=[1,self.num_steps,1,1],
                strides=[1,1,1,1],
                padding='VALID',
                name='pool-2'
                )
        pool_flat_2 = tf.reshape(pooled_2, [-1,self.hidden_size*2])

        trans_out_3 = tf.transpose(outputs_3, perm=[1,0,2])
        for time_step in range(self.num_steps):
            cell_out = tf.reshape(outputs_3[time_step], [-1,self.hidden_size*2])
            att = tf.reduce_sum(cell_out * last_output_1, 1)
            att = tf.reshape(att,[-1,1])
            att_2.append(att)

        s_att_2 = tf.concat(1, att_2)
        s_att_2 = tf.reshape(s_att_2, [-1,self.num_steps,1])
        output_att_3 = trans_out_3 * s_att_2
        pool_reshape_3 = tf.reshape(output_att_3, [-1, self.num_steps, self.hidden_size*2, 1])

        pooled_3 = tf.nn.max_pool(
                pool_reshape_3,
                ksize=[1,self.num_steps,1,1],
                strides=[1,1,1,1],
                padding='VALID',
                name='pool-3'
                )
        pool_flat_3 = tf.reshape(pooled_3, [-1,self.hidden_size*2])
        
        conv_mul_12 = tf.reduce_sum(tf.mul(pool_flat_1, pool_flat_2), 1)
        conv_mul_13 = tf.reduce_sum(tf.mul(pool_flat_1, pool_flat_3), 1)
       
        conv_len_1 = tf.sqrt(tf.reduce_sum(tf.mul(pool_flat_1, pool_flat_1), 1))
        conv_len_2 = tf.sqrt(tf.reduce_sum(tf.mul(pool_flat_2, pool_flat_2), 1))
        conv_len_3 = tf.sqrt(tf.reduce_sum(tf.mul(pool_flat_3, pool_flat_3), 1))

        with tf.name_scope("output"):
            self.cos_12 = tf.div(conv_mul_12, tf.mul(conv_len_1, conv_len_2), name='scores')
            self.cos_13 = tf.div(conv_mul_13, tf.mul(conv_len_1, conv_len_3))

        zero = tf.constant(0, shape=[batch_size], dtype=tf.float32)
        margin = tf.constant(0.2, shape=[batch_size], dtype=tf.float32)
        with tf.name_scope('loss'):
            self.losses = tf.maximum(zero, tf.sub(margin, tf.sub(self.cos_12, self.cos_13)))
            self.loss = tf.reduce_sum(self.losses) + l2_reg_lambda * l2_loss
            print('loss ', self.loss)

        with tf.name_scope('accuracy'):
            self.correct = tf.equal(zero, self.losses)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")

            
 


            
                                                
