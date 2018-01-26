import tensorflow as tf

t1 = tf.ones([2,3,4], dtype=tf.float32)
t2 = tf.ones([2,2,3,4], dtype=tf.float32)

t1_rank = t1.get_shape().ndims
t2_rank = t2.get_shape().ndims

axes = [[t1_rank-1], [t2_rank-1]]
out = tf.tensordot(t1, t2, axes)
print(out)

with tf.Session() as sess:
    sess.run(out)
