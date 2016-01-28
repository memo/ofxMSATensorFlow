# ==============================================================================
### IF YOU ARE RUNNING THIS IN SPYDER MAKE SURE TO USE A NEW CONSOLE EACH TIME
### TO CLEAR THE SESSION
### (press F6, and select 'Execute in a new dedicated Python console')

# ==============================================================================
# Simple script to generate & export tensorflow graph calculating c:=a*b
# based on https://medium.com/jim-fleming/loading-a-tensorflow-graph-with-the-c-api-4caaff88463f#.63x5c9hhg
#
# Modified by Memo Akten to demonstrate ofxMSATensorFlow
# http://github.com/memo/ofxMSATensorFlow
# ==============================================================================



import tensorflow as tf
import shutil
import os

out_path = '../data/models'
out_fname = 'model.pb'

with tf.Session() as sess:
    a = tf.Variable(3.0, name='a')
    b = tf.Variable(4.0, name='b')
    c = tf.mul(a, b, name="c")
    
    sess.run(tf.initialize_all_variables())

    print a.eval()
    print b.eval()
    print c.eval()
    
    # Delete output folder if it exists   
    if os.path.exists(out_path):
        shutil.rmtree(out_path) 
    
    # Write flow graph to disk
    # NB. this won't save value of variables (yet?), see MNIST example
    tf.train.write_graph(sess.graph_def, out_path, out_fname, as_text=False)

