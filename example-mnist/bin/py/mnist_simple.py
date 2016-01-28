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

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
import input_data

import tensorflow as tf
import shutil
import os

out_path = '../data/model-simple'
out_fname = 'mnist-simple'

# Get data
mnist = input_data.read_data_sets("training_data/", one_hot=True)

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
    x = tf.placeholder(tf.float32, [None, 784], name='x_inputs')
    W = tf.Variable(tf.zeros([784, 10]), name="weights_VIZ")
    b = tf.Variable(tf.zeros([10]), name="biases")
    y = tf.nn.softmax(tf.matmul(x, W) + b, name='y_outputs')
    
    
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


    # Init
    sess.run(tf.initialize_all_variables())

    
    # Train
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, {x: batch_xs, y_: batch_ys})

    
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Training complete. Accuracy:", accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))

    save(out_path, out_fname, sess)
    
    # Save variables ? can't load them from C++ :(
    saver = tf.train.Saver()
    saver.save(sess, out_path + '/' + out_fname + '.ckpt')

