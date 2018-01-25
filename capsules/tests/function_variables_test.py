import tensorflow as tf
import numpy as np

def fn1(inputs1):
    inputs = tf.cast(inputs1, tf.float32)
    with tf.variable_scope('conv1') as scope:
        conv1 = tf.contrib.layers.conv2d(
            inputs=inputs,
            num_outputs=1,
            kernel_size=[3,3]
        )
    return conv1

a = tf.placeholder(dtype=tf.float32, shape=[1,9,9,1])
b = fn1(a)

def fn2(inputs):
    a = tf.Variable(tf.zeros(shape=[5,5]), name='a')
    return a

if __name__ == "__main__":
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ar = np.random.rand(1,9,9,1)
        sess.run(fn1(b))
        sess.run(fn2(ar))
