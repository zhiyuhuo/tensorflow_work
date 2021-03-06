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
from data_input_2 import get_lm_input_data 


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
    """
    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    
    def __init__(self, max_seq_len=10):
        self.chunk = []
        self.words = []
        self.labels = []
        self.seqlen = []
        n_samples = 818
        chunk_in, words_in, all_in, rm_out, obj_out, ref_out, dir_out, tar_out, len_words = get_lm_input_data(n_samples)
        print(len(chunk_in))
        print(len(words_in))
        print(len(len_words))
        print(len(rm_out))
        
        print(chunk_in[1])
        print(words_in[1])
        print(len_words[1])
        print(rm_out[1])
        
        for n in range(n_samples):
	    self.chunk.append(chunk_in[n])
            self.words.append(words_in[n])
            self.labels.append(rm_out[n])
            self.seqlen.append(len_words[n])
        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        L = len(self.words)
        if self.batch_id == L:
            self.batch_id = 0
        batch_chunk = (self.chunk[self.batch_id:min(self.batch_id +
                                                  batch_size, L)])
        batch_words = (self.words[self.batch_id:min(self.batch_id +
                                                  batch_size, L)])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, L)])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, L)])
        self.batch_id = min(self.batch_id + batch_size, L)
        return batch_chunk, batch_words, batch_labels, batch_seqlen


# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.005
training_iters = 50000
batch_size = 16
display_step = 10

# Network Parameters
n_chunk = 7
n_words = 66
seq_max_len = 10 # Sequence max length
n_hidden = 128 # hidden layer num of features
n_classes = 2 # linear sequence or not

trainset = LanguageSequenceData(seq_max_len)
testset = LanguageSequenceData(seq_max_len)

# tf Graph input
x_chunk = tf.placeholder("float", [None, n_chunk])
x_words = tf.placeholder("float", [None, seq_max_len, n_words])
y = tf.placeholder("float", [None, n_classes])
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'lstm': tf.Variable(tf.random_normal([n_hidden, n_classes])),
    'chunk': tf.Variable(tf.random_normal([n_chunk, n_classes])),
    'out': tf.Variable(tf.random_normal([n_classes + n_classes, n_classes])),
}
biases = {
    'lstm': tf.Variable(tf.random_normal([n_classes])),
    'chunk': tf.Variable(tf.random_normal([n_classes])),
    'out': tf.Variable(tf.random_normal([n_classes])),
}


def dynamicRNN(x, seqlen, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_words])
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
    return tf.matmul(outputs, weights['lstm']) + biases['lstm']

words_pred = dynamicRNN(x_words, seqlen, weights, biases)
chunks_pred = tf.matmul(x_chunk, weights['chunk']) + biases['chunk']

pred = chunks_pred
#pred = tf.matmul(tf.concat(1, [words_pred, chunks_pred]), weights['out']) + biases['out']


# Define loss and optimizer
cost = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(pred, y)))) 
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

def evaluation(output, target):
    res = 0
    if output[0] > 0.5 and output[1] < 0.5:
        if target[0] > 0.5 and target[1] < 0.5:
	    res = 1	    
    if output[1] > 0.5 and output[0] < 0.5:
        if target[1] > 0.5 and target[0] < 0.5:
	    res = 1	    
    if output[0] < 0.5 and output[1] < 0.5:
        if target[0] < 0.5 and target[1] < 0.5:
	    res = 1
    else:
        res = 0
    return res

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_chunk, batch_words, batch_y, batch_seqlen = trainset.next(batch_size)
        
        #show the data
        #print(batch_x[0])
        #print(batch_y[0])
        #print(batch_seqlen[0])
        
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x_chunk: batch_chunk, x_words: batch_words, y: batch_y, seqlen: batch_seqlen})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x_chunk: batch_chunk, x_words: batch_words, 
						y: batch_y, seqlen: batch_seqlen})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x_chunk: batch_chunk, x_words: batch_words, 
						y: batch_y, seqlen: batch_seqlen})
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
        sess.run(accuracy, feed_dict={x_chunk: batch_chunk, x_words: batch_words, 
						y: batch_y, seqlen: batch_seqlen}))
