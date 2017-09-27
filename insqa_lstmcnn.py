import tensorflow as tf
import numpy as np

class InsConvLSTM(object):
    def __init__(
        self, vectors,sequence_length, batch_size,
        vocab_size, embedding_size,
        filter_sizes, num_filters, l2_reg_lambda=0.0):

        self.input_x_1 = tf.placeholder(tf.int32, [batch_size,sequence_length],name="input_x_1")
        self.input_x_2 = tf.placeholder(tf.int32, [batch_size,sequence_length],name="input_x_2")
        self.input_x_3 = tf.placeholder(tf.int32, [batch_size,sequence_length],name="input_x_3")

        self.hidden_size = 150
        self.num_layers = 2
        self.num_steps = 201
        
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

            self.embedded_chars_1 = chars_1
            self.embedded_chars_2 = chars_2
            self.embedded_chars_3 = chars_3
        self.embedded_chars_expanded_1 = tf.expand_dims(self.embedded_chars_1, -1)
        self.embedded_chars_expanded_2 = tf.expand_dims(self.embedded_chars_2, -1)
        self.embedded_chars_expanded_3 = tf.expand_dims(self.embedded_chars_3, -1)

        conv_output_1 = []
        conv_output_2 = []
        conv_output_3 = []

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded_1,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv-1'
                    )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-1")
                
                zero_1 = tf.zeros([batch_size,filter_size, 1, num_filters])
                cat_1 = tf.concat(1, [conv, zero_1])
                conv_output_1.append(cat_1)
                
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded_2,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv-2'
                    )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-2")

           
                zero_2 = tf.zeros([batch_size, filter_size, 1, num_filters])
                cat_2 = tf.concat(1, [conv, zero_2])
                conv_output_2.append(cat_2)

                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded_3,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv-3'
                    )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-3")
                
                zero_3 = tf.zeros([batch_size, filter_size, 1, num_filters])
                cat_3 = tf.concat(1, [conv, zero_3])
                conv_output_3.append(cat_3)

        num_filters_total = num_filters * len(filter_sizes)
        
        self.conv_output_1 = conv_output_1

        conv_reshape_1 = tf.reshape(tf.concat(3, conv_output_1), [-1,num_filters_total])
        conv_reshape_2 = tf.reshape(tf.concat(3, conv_output_2), [-1,num_filters_total])
        conv_reshape_3 = tf.reshape(tf.concat(3, conv_output_3), [-1,num_filters_total])

        self.conv_reshape_1 = conv_reshape_1


        with tf.variable_scope("Medium") as scope:
            W = tf.get_variable("W", shape=[num_filters_total, 200], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[200]), name="b")
            tanh_output_1 = tf.nn.tanh(tf.nn.xw_plus_b(conv_reshape_1, W, b))
            tanh_output_2 = tf.nn.tanh(tf.nn.xw_plus_b(conv_reshape_2, W, b))
            tanh_output_3 = tf.nn.tanh(tf.nn.xw_plus_b(conv_reshape_3, W, b))

        reshape_output_1 = tf.reshape(tanh_output_1, [-1, 201, 200])
        reshape_output_2 = tf.reshape(tanh_output_2, [-1, 201, 200])
        reshape_output_3 = tf.reshape(tanh_output_3, [-1, 201, 200])
        
        
        #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True)
        #cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers, state_is_tuple=True)

        output_1 = []
        output_2 = []
        output_3 = []
        #state = self._initial_state

        with tf.variable_scope("RNN-1") as scope:
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers, state_is_tuple=True)
            state = cell.zero_state(batch_size, tf.float32)
            for time_step in range(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(reshape_output_1[:, time_step, :], state)
                re_cell = tf.reshape(cell_output, [-1,1,self.hidden_size])
                output_1.append(re_cell)

            state = cell.zero_state(batch_size, tf.float32)
            for time_step in range(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(reshape_output_2[:, time_step, :], state)
                re_cell = tf.reshape(cell_output, [-1,1,self.hidden_size])
                output_2.append(re_cell)

            state = cell.zero_state(batch_size, tf.float32)
            for time_step in range(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(reshape_output_3[:, time_step, :], state)
                re_cell = tf.reshape(cell_output, [-1,1,self.hidden_size])
                output_3.append(cell_output)
    

        pool_reshape_1 = tf.reshape(tf.concat(1,output_1),[-1,self.num_steps,self.hidden_size,1])
        pool_reshape_2 = tf.reshape(tf.concat(1,output_2),[-1,self.num_steps,self.hidden_size,1])
        pool_reshape_3 = tf.reshape(tf.concat(1,output_3),[-1,self.num_steps,self.hidden_size,1])
               
        pooled_1 = tf.nn.max_pool(
            pool_reshape_1,
            #ksize=[1, sequence_length - filter_size + 1, 1, 1],
            ksize=[1, self.num_steps, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name='pool-1'
            )
          
        pooled_2 = tf.nn.max_pool(
            pool_reshape_2,
            ksize=[1, self.num_steps, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name='pool-2',
            )
        pooled_3 = tf.nn.max_pool(
            pool_reshape_3,
            ksize=[1, self.num_steps, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name='pool-3',
            )


        pool_flat_1 = tf.reshape(pooled_1,[-1,self.hidden_size])
        pool_flat_2 = tf.reshape(pooled_2,[-1,self.hidden_size])
        pool_flat_3 = tf.reshape(pooled_3,[-1,self.hidden_size])
        self.conv_flat_1 = pool_flat_1

        conv_mul_12 = tf.reduce_sum(tf.mul(pool_flat_1, pool_flat_2), 1)
        conv_mul_13 = tf.reduce_sum(tf.mul(pool_flat_1, pool_flat_3), 1)

        conv_len_1 = tf.sqrt(tf.reduce_sum(tf.mul(pool_flat_1, pool_flat_1), 1))
        conv_len_2 = tf.sqrt(tf.reduce_sum(tf.mul(pool_flat_2, pool_flat_2), 1))
        conv_len_3 = tf.sqrt(tf.reduce_sum(tf.mul(pool_flat_3, pool_flat_3), 1))

        self.mul_13 = tf.abs(conv_mul_13)
        self.len_3 = conv_len_3
        with tf.name_scope("output"):
            self.cos_12 = tf.div(conv_mul_12, tf.mul(conv_len_1, conv_len_2), name="scores")
            self.cos_13 = tf.div(conv_mul_13, tf.mul(conv_len_1, conv_len_3))
            

        zero = tf.constant(0, shape=[batch_size], dtype=tf.float32)
        margin = tf.constant(0.2, shape=[batch_size], dtype=tf.float32)
        with tf.name_scope("loss"):
            self.losses = tf.maximum(zero, tf.sub(margin, tf.sub(self.cos_12, self.cos_13)))
            #self.losses = tf.maximum(zero, tf.sub(margin, self.cos_12))
            self.loss = tf.reduce_sum(self.losses) + l2_reg_lambda * l2_loss
            print('loss ', self.loss)

        with tf.name_scope("accuracy"):
            self.correct = tf.equal(zero, self.losses)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")
            

                

                
                    
                
            
