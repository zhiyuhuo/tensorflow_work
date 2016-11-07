'''
A Dynamic Recurrent Neural Network (LSTM) implementation example using
TensorFlow library. This program will use drnn on the spatial language data to build a classifier on spatial clauses

Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import random
from data_input import get_lm_input_data 


# ====================
#  import binary language feature data
# ====================
class LanguageSequenceData(object):
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:

    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:
    - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])

    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    
    def __init__(self, max_seq_len=10):
        self.data = []
        self.labels = []
        self.seqlen = []
        n_samples = 818
        tuple_mat_chunk_input, tuple_mat_words_input, tuple_mat_label_output, len_mat_words_input, tuple_mat_join_input = get_lm_input_data(n_samples)
        #print(len(tuple_mat_join_input))
        #print(len(tuple_mat_words_input))
        #print(len(tuple_mat_label_output))
        #print(tuple_mat_join_input[0])
        #print(len((tuple_mat_join_input[0])[0]))
        #exit()
        
        for n in range(n_samples):
	    self.seqlen.append(len_mat_words_input[n])
            self.data.append(tuple_mat_join_input[n])
            self.labels.append(tuple_mat_label_output[n])
        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen


# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.001
training_iters = 500000
batch_size = 1000
display_step = 10

# Network Parameters
n_dim = 73
seq_max_len = 10 # Sequence max length
n_hidden = 73 # hidden layer num of features
n_classes = 2 # linear sequence or not

trainset = LanguageSequenceData(seq_max_len)
testset = LanguageSequenceData(seq_max_len)

# tf Graph input
x = tf.placeholder("float", [None, seq_max_len, n_dim])
y = tf.placeholder("float", [None, n_classes])
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
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

# Define loss and optimizer
cost = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(pred, y)))) 
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

def classify(output, target):
    res = 0
    if output[0] > 0.5 && output[1] < 0.5:
        if target[0] > 0.5 && target[1] < 0.5:
	    res = 1	    
    else if output[1] > 0.5 && output[0] < 0.5:
        if target[1] > 0.5 && target[0] < 0.5:
	    res = 1	    
    else if output[0] < 0.5 && output[1] < 0.5:
        if target[0] < 0.5 && target[1] < 0.5:
	    res = 1
    else
        res = 0
    return res

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
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,
                                                seqlen: batch_seqlen})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y,
                                             seqlen: batch_seqlen})
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
        sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                      seqlen: test_seqlen}))
