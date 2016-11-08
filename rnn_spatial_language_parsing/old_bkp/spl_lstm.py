#Inspired by https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3%20-%20Neural%20Networks/recurrent_network.py
import tensorflow as tf

import numpy as np
from data_input import get_lm_input_data

# configuration
#                        O * W + b -> 10 labels for each image, O[? 28], W[28 10], B[10]
#                       ^ (O: output 28 vec from 28 vec input)
#                       |
#      +-+  +-+       +--+
#      |1|->|2|-> ... |28| time_step_size = 28
#      +-+  +-+       +--+
#       ^    ^    ...  ^
#       |    |         |
# img1:[28] [28]  ... [28]
# img2:[28] [28]  ... [28]
# img3:[28] [28]  ... [28]
# ...
# img128 or img256 (batch_size or test_size 256)
#      each input size = input_vec_size=lstm_size=28

# configuration variables
N = 818
input_vec_size = lstm_size = 66
label_size = 20
time_step_size = 1

tuple_mat_chunk_input, tuple_mat_words_input, tuple_mat_label_output = get_lm_input_data(818)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, W, B, lstm_size, time_step_size):
    # X, input shape: (batch_size, time_step_size, input_vec_size)
    #XT = tf.transpose(X)  # permute time_step_size and batch_size
    # X shape: (time_step_size * batch_size, input_vec_size)
    X_in = tf.split(0, time_step_size, X) # split them to time_step_size
    # Each array shape: (batch_size, input_vec_size)

    # Make lstm with lstm_size (each input vector size)
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)

    # Get lstm cell output, time_step_size (28) arrays with lstm_size output: (batch_size, lstm_size)
    outputs, _states = tf.nn.rnn(lstm, X_in, dtype=tf.float32)

    # Linear activation
    # Get the last output
    return tf.matmul(outputs[-1], W) + B, lstm.state_size # State size to initialize the stat

X = tf.placeholder("float", [None, input_vec_size])
Y = tf.placeholder("float", [label_size])

## get lstm_size and output 10 labels
W = init_weights([lstm_size, label_size])
B = init_weights([label_size])

py_x, state_size = model(X, W, B, lstm_size, time_step_size)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

## Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(100):
        for n in range(N):
	    trX = tuple_mat_words_input[n]
	    trY = tuple_mat_label_output[n]
	    print(trX)
	    print(trY)
	    time_step_size = len(trX)
            sess.run(train_op, feed_dict={X: trX, Y: trY})

        #test_indices = np.arange(len(teX))  # Get A Test Batch
        #np.random.shuffle(test_indices)
        #test_indices = test_indices[0:test_size]

        #print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         #sess.run(predict_op, feed_dict={X: teX[test_indices]})))
