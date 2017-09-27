import tensorflow as tf
import numpy

class InsAttLSTM(object):
    def __init__(
        self, vectors, sequence_length, batch_size,
        vocab_size, embedding_size,
        filter_sizes, num_filters, l2_reg_lambda=0.0):

        self.input_x_1 = tf.placeholder(tf.int32, [batch_size, sequence_length],name='input_x_1')
        self.input_x_2 = tf.placeholder(tf.int32, [batch_size, sequence_length],name="input_x_2")
        self.input_x_3 = tf.placeholder(tf.int32, [batch_size, sequence_length],name="input_x_3")

        self.hidden_size = 100
        self.num_layers = 1
        self.num_steps = 200

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

        output_1 = []
        output_2 = []
        output_3 = []
        att_1 = []
        att_2 = []
        
        with tf.variable_scope("RNN-1") as scope:
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0,state_is_tuple=True)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*self.num_layers, state_is_tuple=True)
            '''
            Wa = tf.Variable(
                tf.random_uniform([self.hidden_size, 10], -1.0, 1.0),
                name='Wa')
            Wq = tf.Variable(
                tf.random_uniform([self.hidden_size, 10], -1.0, 1.0),
                name='Wq')
            
            wm = tf.Variable(
                tf.random_uniform([10,1], -1.0,1.0),
                name='wm')
            '''
            Wa = tf.Variable(
                tf.random_uniform([1, self.hidden_size], -1.0, 1.0),
                name='Wa')
            Wq = tf.Variable(
                tf.random_uniform([1, self.hidden_size], -1.0, 1.0),
                name='Wq')
            wm = tf.Variable(
                tf.random_uniform([1,1],-1.0,1.0),
                name='wm')
            
            state = cell.zero_state(batch_size, tf.float32)
            for time_step in range(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(chars_1[:, time_step, :], state)
                re_cell = tf.reshape(cell_output, [-1,1,self.hidden_size])
                output_1.append(re_cell)

            pool_reshape_1 = tf.reshape(tf.concat(1,output_1),[-1,self.num_steps,self.hidden_size,1])
            pooled_1 = tf.nn.max_pool(
                pool_reshape_1,
                ksize=[1, self.num_steps, 1, 1],
                strides=[1,1,1,1],
                padding='VALID',
                name='pool-1'
                )
            pool_flat_1 = tf.reshape(pooled_1, [-1,self.hidden_size])
            #print pooled_1.get_shape().as_list()
            
            
            state = cell.zero_state(batch_size, tf.float32)
            
            for time_step in range(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(chars_2[:, time_step, :], state)
                re_cell = tf.reshape(cell_output, [-1,1,self.hidden_size])
                #print cell_output.get_shape().as_list()

                mq = tf.matmul(pool_flat_1,tf.matrix_transpose(Wq))+tf.matmul(cell_output,tf.matrix_transpose(Wa))
                ms = tf.matmul(tf.tanh(mq), wm)
                ms = tf.reshape(ms,[-1,1])
                att_1.append(ms)
                output_2.append(re_cell)

            output_2 = tf.reshape(tf.concat(1,output_2),[-1,self.num_steps,self.hidden_size])
            #s_att_1 = tf.nn.softmax(tf.concat(1, att_1),dim=-1)
            s_att_1 = tf.concat(1, att_1)
            s_att_1 = tf.reshape(s_att_1, [-1, self.num_steps, 1])
            output_att_2 = output_2 * s_att_1
            pool_reshape_2 = tf.reshape(output_att_2, [-1, self.num_steps, self.hidden_size,1])

            pooled_2 = tf.nn.max_pool(
                pool_reshape_2,
                ksize=[1,self.num_steps,1,1],
                strides=[1,1,1,1],
                padding='VALID',
                name='pool-2'
                )
            pool_flat_2 = tf.reshape(pooled_2, [-1,self.hidden_size])
            #pool_flat_2 = tf.reshape(pooled_2, [-1,1])
            

            state = cell.zero_state(batch_size, tf.float32)
            for time_step in range(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(chars_3[:, time_step, :], state)
                re_cell = tf.reshape(cell_output, [-1,1,self.hidden_size])

                mq = tf.matmul(pool_flat_1,tf.matrix_transpose(Wq))+tf.matmul(cell_output,tf.matrix_transpose(Wa))

                ms = tf.matmul(tf.tanh(mq),wm)
                ms = tf.reshape(ms,[-1,1])
                att_2.append(ms)
                output_3.append(re_cell)

            output_3 = tf.reshape(tf.concat(1, output_3),[-1,self.num_steps,self.hidden_size])
            #print (tf.concat(1,att_1)).get_shape().as_list()
            
            #s_att_2 = tf.nn.softmax(tf.concat(1,att_2),dim=-1)
            s_att_2 = tf.concat(1,att_2)
            s_att_2 = tf.reshape(s_att_2, [-1, self.num_steps, 1])
            output_att_3 = output_3 * s_att_2
            pool_reshape_3 = tf.reshape(output_att_3, [-1, self.num_steps, self.hidden_size,1])

            pooled_3 = tf.nn.max_pool(
                pool_reshape_3,
                ksize=[1,self.num_steps,1,1],
                strides=[1,1,1,1],
                padding='VALID',
                name='pool-3'
                )
            pool_flat_3 = tf.reshape(pooled_3, [-1,self.hidden_size])

        self.att1 = s_att_1
        self.att2 = s_att_2
        conv_mul_12 = tf.reduce_sum(tf.mul(pool_flat_1, pool_flat_2), 1)
        conv_mul_13 = tf.reduce_sum(tf.mul(pool_flat_1, pool_flat_3), 1)

        conv_len_1 = tf.sqrt(tf.reduce_sum(tf.mul(pool_flat_1, pool_flat_1), 1))
        conv_len_2 = tf.sqrt(tf.reduce_sum(tf.mul(pool_flat_2, pool_flat_2), 1))
        conv_len_3 = tf.sqrt(tf.reduce_sum(tf.mul(pool_flat_3, pool_flat_3), 1))

        self.mul_12 = output_2
        self.mul_13 = output_att_2
        self.len_2 = s_att_2
        self.len_3 = output_att_3
        with tf.name_scope("output"):
            self.cos_12 = tf.div(conv_mul_12, tf.mul(conv_len_1, conv_len_2), name="scores")
            self.cos_13 = tf.div(conv_mul_13, tf.mul(conv_len_1, conv_len_3))
            

        zero = tf.constant(0, shape=[batch_size], dtype=tf.float32)
        margin = tf.constant(0.2, shape=[batch_size], dtype=tf.float32)
        with tf.name_scope("loss"):
            self.losses = tf.maximum(zero, tf.sub(margin, tf.sub(self.cos_12, self.cos_13)))
            #self.losses = tf.maximum(zero, tf.sub(margin, self.cos_12))
            self.loss = tf.reduce_sum(self.losses) + l2_reg_lambda * l2_loss
            #print('loss ', self.loss)

        with tf.name_scope("accuracy"):
            self.correct = tf.equal(zero, self.losses)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")
            


            
                                                
