#!/usr/bin/env python
import tensorflow as tf
import numpy as np
from triddl_input import read_data

D = 200
C = 5
D_hidden = 150
BATCH = 1
CORRUPTION_LEVEL = 0.1

# create node for input data
X = tf.placeholder("float", [None, D], name='X')
# create node for corruption mask (for robust autoencoder)
mask = tf.placeholder("float", [None, D], name='mask')

# create nodes for hidden variables
W_init_max = 4 * np.sqrt(6. / (D + D_hidden))
W_init = tf.random_uniform(shape=[D, D_hidden], minval=-W_init_max, maxval=W_init_max)

#encoder W, b
W = tf.Variable(W_init, name='W')
b = tf.Variable(tf.zeros([D_hidden]), name='b')

#decoder W, b
W_prime = tf.transpose(W)  # tied weights between encoder and decoder
b_prime = tf.Variable(tf.zeros([D]), name='b_prime')

def model(X, mask, W, b, W_prime, b_prime):
    tilde_X = mask * X  # corrupted X

    Y = tf.nn.sigmoid(tf.matmul(tilde_X, W) + b)  # hidden state
    Z = tf.nn.sigmoid(tf.matmul(Y, W_prime) + b_prime)  # reconstructed input
    return Z
  
# build model graph
Z = model(X, mask, W, b, W_prime, b_prime)

# create cost function
cost = tf.reduce_mean(tf.pow(X - Z, 2))  # minimize squared error
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)  # construct an optimizer
predict_op = Z

# load 3ddl data
trX, trY, teX, teY = read_data('3ddl_data', D, C)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(500):
        for start, end in zip(range(0, len(trX), BATCH), range(BATCH, len(trX)+1, BATCH)):
            input_ = trX[start:end]
            mask_np = np.random.binomial(1, 1 - CORRUPTION_LEVEL, input_.shape)
            sess.run(train_op, feed_dict={X: input_, mask: mask_np})

        mask_np = np.random.binomial(1, 1 - CORRUPTION_LEVEL, teX.shape)
        print(i, sess.run(cost, feed_dict={X: teX, mask: mask_np}))
    # save the predictions for 100 images
    mask_np = np.random.binomial(1, 1 - CORRUPTION_LEVEL, teX[:100].shape)
    predicted_imgs = sess.run(predict_op, feed_dict={X: teX[:100], mask: mask_np})
    input_vec = teX[:100]
    
    # show the encoders
    print W
    print b


