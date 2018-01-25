import tensorflow as tf
import numpy as np

a = tf.placeholder(dtype=tf.float32, shape=[None,2])
b = tf.placeholder(dtype=tf.float32, shape=[None,2])
c = tf.stack([a,b], 0)
d = tf.shape(c)


if __name__ == "__main__":
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ar = np.random.rand(10,2)
        print(sess.run([c, d], feed_dict={a: ar, b: ar}))
