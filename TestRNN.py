import tensorflow as tf
import numpy as np

n_inputs = 3
n_neurons = 5

X0 = tf.placeholder(tf.float32,shape=[None,n_inputs])
X1 = tf.placeholder(tf.float32,shape=[None,n_inputs])

Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons]),dtype=tf.float32)
Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons]),dtype=tf.float32)
b = tf.Variable(tf.zeros(shape=[1, n_neurons]), dtype=tf.float32)

Y0 = tf.tanh(tf.matmul(X0, Wx)+ b)
Y1 = tf.tanh(tf.matmul(X1, Wx)+ tf.matmul(Y0,Wy) + b)

init = tf.global_variables_initializer()

X0_batch = np.array([[0,1,2], [3,4,5], [6,7,8], [9,0,1]])
X1_batch = np.array([[9,8,7], [0,0,0], [6,5,4], [3,2,1]])

with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1:X1_batch})
    print(Y0_val)
    print(Y0_val.shape)
    print(type(Y0_val))
    print(Y1_val)
    print(Y1_val.shape)
    print(type(Y1_val))



