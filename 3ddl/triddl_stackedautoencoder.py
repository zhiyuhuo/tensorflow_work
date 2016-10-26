# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on 3ddl dataset for furniture recognition.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from triddl_input import read_data

# Parameters
learning_rate = 0.01
training_epochs = 1000
batch_size = 1
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 128 # 1st layer num features
n_hidden_2 = 64 # 2nd layer num features
n_hidden_3 = 32 # 3nd layer num features
n_input = 200 # 3ddl data input (shape: 200*1 vector)
n_class = 5

# import the 3ddl data
trX, trY, teX, teY = read_data('3ddl_data', n_input, n_class)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_class])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
    'softmax':    tf.Variable(tf.random_normal([n_hidden_3, n_class])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b3': tf.Variable(tf.random_normal([n_input])),
    'softmax':    tf.Variable(tf.random_normal([n_class])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    # Decoder Hidden layer with sigmoid activation #3
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    return layer_3


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    # Decoder Hidden layer with sigmoid activation #3
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   biases['decoder_b3']))
    return layer_3
  
# Building softmax classification
def classify(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    # Decoder Hidden layer with sigmoid activation #3
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    # Classification Result layer with sigmoid activation 
    res     = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['softmax']),
				   biases['softmax']))
    return res
  
# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)
output = classify(X)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer of the autoencoder, minimize the squared error
cost_autoencoder = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer_autoencoder = tf.train.RMSPropOptimizer(learning_rate).minimize(cost_autoencoder)

# Define loss and optimizer of the softmax classification, minimize the squared error
cost_softmax = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(output), reduction_indices=1))
optimizer_softmax = tf.train.RMSPropOptimizer(learning_rate).minimize(cost_softmax)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph and train the auto encoder
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size)):
            input_ = trX[start:end]
	    _, c = sess.run([optimizer_autoencoder, cost_autoencoder], feed_dict={X: input_})
        
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost_autoencoder=", "{:.9f}".format(c))

    print("Optimization Finished!")

    for epoch in range(training_epochs):
        # loop over all batches
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size)):
            input_ = trX[start:end]
            target_ = trY[start:end]
	    _, c = sess.run([optimizer_autoencoder, cost_autoencoder], feed_dict={X: input_, Y: target_ })
        
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost_autoencoder=", "{:.9f}".format(c))
        
    # Test the model
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
    print("Accuracy:", accuracy.eval({X: teX, Y: teY}))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    