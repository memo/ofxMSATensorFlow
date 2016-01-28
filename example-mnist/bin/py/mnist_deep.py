# ==============================================================================
### IF YOU ARE RUNNING THIS IN SPYDER MAKE SURE TO USE A NEW CONSOLE EACH TIME
### TO CLEAR THE SESSION
### (press F6, and select 'Execute in a new dedicated Python console')

# ==============================================================================
# Copyright 2015 Google Inc. All Rights Reserved.
# Modified by Memo Akten to demonstrate ofxMSATensorFlow
# http://github.com/memo/ofxMSATensorFlow
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Convolution neural network mnist classifier, based on
https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/pros/index.html
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
import input_data

import tensorflow as tf
import shutil
import os

out_path = '../data/model-deep'
out_fname = 'mnist-deep'
train_file ='ckpt-deep/'+out_fname + '.ckpt'

# Get data
mnist = input_data.read_data_sets("training_data/", one_hot=True)


def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)
  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def save(path, fname, sess):
    # Write flow graph to disk
    # MEGA UGLY HACK BECAUSE TENSOR FLOW DOESN'T SAVE VARIABLES
    # NB. tf.train.write_graph won't save value of variables (yet?)
    # so need to save value of variables as constants, 
    # and then in C++ push the constants back to the vars :S
    # based on https://stackoverflow.com/questions/34343259/is-there-an-example-on-how-to-generate-protobuf-files-holding-trained-tensorflow/34343517

    for v in tf.all_variables():
        vc = tf.constant(v.eval(sess))
        n = v.name.split(":")[0]    # get name (not sure what the :0 is)
        tf.assign(v, vc, name=n+"_VARHACK")        
        
    # Delete output folder if it exists    
    if os.path.exists(out_path):
        shutil.rmtree(out_path)   
    
    fname = fname+".pb";
    print("Saving to ", path+"/"+fname, "...")
    tf.train.write_graph(sess.graph_def, path, fname, as_text=False)        
    print("...done.")


with tf.Session() as sess:
    # Create the model
    # As a temp measure I'm adding _VIZ to the names of layers I want to visualize
    
    # inputs
    x = tf.placeholder(tf.float32, [None, 784], name='x_inputs')

    # first conv layer + pool
    W_conv1 = weight_variable([5, 5, 1, 32], "layer1_weights")
    b_conv1 = bias_variable([32], "layer1_biases")
    x_image = tf.reshape(x, [-1,28,28,1])    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second conv layer + pool
    W_conv2 = weight_variable([5, 5, 32, 64], "layer2_weights")
    b_conv2 = bias_variable([64], "layer2_biases")
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    # first fully connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024], "layer3_weights_VIZ")
    b_fc1 = bias_variable([1024], "layer3_biases")
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    # dropout
    keep_prob = tf.placeholder("float", name="keep_prob")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # read out
    W_fc2 = weight_variable([1024, 10], "layer4_weights_VIZ")
    b_fc2 = bias_variable([10], "layer4_biases")
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y_outputs')

    # desired output
    y_ = tf.placeholder(tf.float32, [None, 10])
    
    # calculate loss
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.initialize_all_variables())

    # Saver, to save/load training progress 
    # Can't loads this in C++ (shame), only useful for contining training
    saver = tf.train.Saver()    

    #comment out this line to start training from the beginning
    saver.restore(sess, train_file); 
    
    for i in range(5000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            saver.save(sess, train_file)            
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step", i, "training accuracy %", train_accuracy)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    
    print("test accuracy", accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    save(out_path, out_fname, sess)

 

