import tensorflow as tf
import numpy as np

t1 = tf.ones([2,3,4], dtype=tf.float32)
t2 = tf.ones([2,2,3,4], dtype=tf.float32)

t1_rank = t1.get_shape().ndims
t2_rank = t2.get_shape().ndims

axes = [[t1_rank-1], [t2_rank-1]]
out = tf.tensordot(t1, t2, axes)
print(out)


t1 = tf.placeholder(dtype=tf.float32, shape=[None, 3, 6, 5])
t2 = tf.placeholder(dtype=tf.float32, shape=[3, 4, 5])

i0 = tf.constant(0)

def tensordot(tup):
    t1 = tup[0]
    t2 = tup[1]
    d1 = t1.get_shape().ndims - 1
    d2 = t2.get_shape().ndims - 1
    return tf.tensordot(t1, t2, [[d1], [d2]])

def nested_dot(t1, t2):
    elems = (t1, t2)
    return tf.map_fn(lambda x: tensordot(x), elems, tf.float32)

res1 = tf.map_fn(
    fn = lambda x: nested_dot(x, t2),
    elems = t1,
    dtype = tf.float32
)

print(res1)
# result2 = tf.while_loop(
#     cond = lambda x: condition(),
#     body = lambda x: body(),
#     loop_vars = [i0, ],
# )



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ar1 = np.random.rand(10, 3, 6, 5)
    ar2 = np.random.rand(3,4,5)
    print(sess.run(res1, {t1: ar1, t2: ar2}))
    # sess.run(out)
