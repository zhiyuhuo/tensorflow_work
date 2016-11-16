'''
A Dynamic Recurrent Neural Network (LSTM) implementation example using
TensorFlow library. This program will use drnn on the spatial language data to build a classifier on spatial clauses

Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy
import random
from data_input import LanguageSequenceData 

# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 32
display_step = 10

# Network Parameters
n_dim = 73
seq_max_len = 10 # Sequence max length
n_hidden = 512 # hidden layer num of features
n_classes = 3 # linear sequence or not

trainset = LanguageSequenceData('train', 'target_room')
testset = LanguageSequenceData('test', 'target_room')

# tf Graph input
x = tf.placeholder("float", [None, seq_max_len, n_dim])
y = tf.placeholder("float", [None, n_classes])
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes])),
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes])),
}


def dynamicRNN(x, seqlen, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_dim])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, seq_max_len, x)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.pack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']

pred = dynamicRNN(x, seqlen, weights, biases)

#pred = tf.matmul(tf.concat(1, [words_pred, chunks_pred]), weights['out']) + biases['out']


# Define loss and optimizer
#cost = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(pred, y)))) 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

_iftrain = 0    
    
if __name__ == "__main__":
    # Initializing the variables
    init = tf.initialize_all_variables()
    
    if _iftrain == 1:
	# Launch the graph
	with tf.Session() as sess:
	    sess.run(init)
	    step = 1
	    # Keep training until reach max iterations
	    while step * batch_size < training_iters:
		batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
		
		#show the data
		#print(batch_x[0])
		#print(batch_y[0])
		#print(batch_seqlen[0])
		
		# Run optimization op (backprop)
		sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
		if step % display_step == 0:
		    # Calculate batch accuracy
		    acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
		    # Calculate batch loss
		    loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
		    print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
			  "{:.6f}".format(loss) + ", Training Accuracy= " + \
			  "{:.5f}".format(acc))
		step += 1
	    print("Optimization Finished!")

	    # Calculate accuracy
	    test_data = testset.data
	    test_label = testset.labels
	    test_seqlen = testset.seqlen
	    print("Testing Accuracy:", \
		sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen}))
	    
	    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
	    save_path = saver.save(sess, "./model_rm.ckpt")
	
    
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2) 
    with tf.Session() as sess:
        batch_x, batch_y, batch_seqlen = testset.next(818)
        print(len(batch_x))
        print(len(batch_y))
        print(len(batch_seqlen))
        
        saver.restore(sess, "./model_rm.ckpt")
        print("Testing Accuracy 1:", sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen}))
        
        res = sess.run(tf.argmax(pred,1), feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
        groundtruth_label = sess.run(tf.argmax(y,1), feed_dict={y: batch_y})
        print("Testing Result:", zip(res, groundtruth_label))
	save_file = open('test_room.out', 'w')
	for r in res:
            save_file.write("%d " % r)
        save_file.write("\n")
        for g in groundtruth_label:
            save_file.write("%d " % g)
        
	
	
	
	
	
	
	
	
	
	
	