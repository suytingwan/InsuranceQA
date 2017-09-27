#! /usr/bin/env python3.4

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import insurance_qa_data_helpers
from insqa_dotlstm import InsBiDotLSTM
import operator
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

#print tf.__version__

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 30, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 10000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("test_every", 20000, "test model")
tf.flags.DEFINE_integer("checkpoint_every", 10000, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
f = open('params.txt','w')

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
    f.write('{}={}'.format(attr.upper(), value))
    f.write('\n')
print("")
f.close()

# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")

vocab = insurance_qa_data_helpers.build_vocab()
alist = insurance_qa_data_helpers.read_alist()
raw = insurance_qa_data_helpers.read_raw()
x_train_1, x_train_2, x_train_3 = insurance_qa_data_helpers.load_data_6(vocab, alist, raw, FLAGS.batch_size)
valList, vectors = insurance_qa_data_helpers.load_val_and_vectors()
testList = insurance_qa_data_helpers.load_test_and_vectors()

print('x_train_1', np.shape(x_train_1))
print("Load done...")

val_file = '../insuranceQA/dev_all.txt'
test_file = '../insuranceQA/test1_all.txt'
#precision = '../insuranceQA/dev_dot.acc'
#x_val, y_val = data_deepqa.load_data_val()

# Training
# ==================================================

with tf.Graph().as_default():
  with tf.device("/gpu:1"):
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = InsBiDotLSTM(
            vectors=vectors,
            sequence_length=x_train_1.shape[1],
            batch_size=FLAGS.batch_size,
            vocab_size=len(vocab),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(0.001)
        #optimizer = tf.train.GradientDescentOptimizer(1e-1)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Output directory for models and summaries
        #timestamp = str(int(time.time()))
        path = 'bi_point1'
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", path))
        precision = os.path.abspath(os.path.join(out_dir,'dev_all.acc'))
        print("Writing to {}\n".format(out_dir))

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())
            sess.run(tf.initialize_all_variables())
        else:
            print 'loading model'
            checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
            #saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver = tf.train.Saver(tf.all_variables())
            saver.restore(sess,checkpoint_file)
            print 'loading model done ...'

        def dev_step(tag):
          scoreList = []
          i = int(0)
          while True:
              if tag == 1:               
                  x_test_1, x_test_2, x_test_3 = insurance_qa_data_helpers.load_data_val_6(valList, vocab, i, FLAGS.batch_size)
              if tag == 2:
                  x_test_1, x_test_2, x_test_3 = insurance_qa_data_helpers.load_data_val_6(testList, vocab, i, FLAGS.batch_size)
              feed_dict = {
                cnn.input_x_1: x_test_1,
                cnn.input_x_2: x_test_2,
                cnn.input_x_3: x_test_3,
                cnn.dropout_keep_prob: 1.0
              }
              batch_scores = sess.run([cnn.cos_12], feed_dict)
              for score in batch_scores[0]:
                  scoreList.append(score)
              i += FLAGS.batch_size
              if i >= len(testList):
                  break

          np.savetxt('valscore.txt',scoreList)
          sessdict = {}
          index = int(0)
          if tag == 1:
              filename = val_file
          elif tag == 2:
              filename = test_file
          for line in open(filename):
              items = line.strip().split(' ')
              qid = items[1].split(':')[1]
              if not qid in sessdict:
                  sessdict[qid] = []
              sessdict[qid].append((scoreList[index], items[0]))
              index += 1
              if index >= len(testList):
                  break
          lev1 = float(0)
          lev0 = float(0)
          #of = open(precision, 'a')

          for k, v in sessdict.items():
              v.sort(key=operator.itemgetter(0), reverse=True)
              score, flag = v[0]
              if flag == '1':
                  lev1 += 1
              if flag == '0':
                  lev0 += 1
          #if tag == 2:
              #of.write('test: '+'\n')
              #print('test:'+'\n')
          #of.write('lev1:' + str(lev1) + '\n')
          #of.write('lev0:' + str(lev0) + '\n')
          print('lev1 ' + str(lev1))
          print('lev0 ' + str(lev0))
          #of.close()

        dev_step(1)
