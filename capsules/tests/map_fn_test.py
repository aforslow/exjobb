import tensorflow as tf
import numpy as np

# def fn(tup):
#     # shape tup = ([2, 3], [4, 2])
#     t1 = tup[0]
#     t2 = tup[1]
#     # shape t1 = [None, 2]
#     # shape t2 = [None, 2]
#     out = tf.tensordot(t1, t2, [[0], [1]])
#     return out
#
def tensor_dot(tup):
    # shape tup = ([?,6,6,8], [16,10,8])
    with tf.variable_scope("caps_dot") as scope:
        t1 = tup[0]
        t2 = tup[1]
        t1_rank = t1.get_shape().ndims
        t2_rank = t2.get_shape().ndims
        axes = [[t1_rank-1], [t2_rank-1]]
        out = tf.tensordot(t1, t2, axes)
        print(out)
    return out

def caps_dot(tup):
    return tf.map_fn(lambda x: tensor_dot(x), tup, dtype=tf.float32)

def nested_caps_dot(tup, i):
    if i == 0:
        return caps_dot(tup)
    else:
        with tf.variable_scope('nested_caps_dot') as scope:
            out = tf.map_fn(lambda x: nested_caps_dot(x, i-1), tup, dtype=tf.float32)
            print(out)
            return out

def trans_caps_dot(tup):
    return tf.map_fn(lambda x: caps_dot(x), tup, dtype=tf.float32)

# tf.map(lambda x: tf.map(lambda y: cap))
# elems = (np.array([1, 2, 3], dtype=np.int64), np.array([-1, 1, -1], dtype=np.int64))
# a = tf.placeholder(tf.float32, [None, 3, 2])
# b = tf.placeholder(tf.float32, [None, 4, 2])
a = tf.ones([32,5,3,2])
b = tf.ones([32,5,4,2])
elems = (a,b)
# alternate = tf.map_fn(lambda x: nested_map(x), elems, dtype=tf.float32)
# alternate = tf.Tensor(nested_caps_dot(elems, 2))
# alternate1 = nested_caps_dot(elems,2)
# alternate = tf.convert_to_tensor(alternate1, dtype=tf.float32)
alternate = trans_caps_dot(elems)
c = tf.ones([32,5,3,4])
d = alternate + c
print(alternate)

a1 = tf.placeholder(tf.float32, [None, 5, 4, 2])
b1 = tf.placeholder(tf.float32, [None, 5, 4, 2])
elems1 = (a1, b1)

def res1fn(x):
    t1 = x[0]
    t2 = x[1]
    out = tf.tensordot(t1, t2, [[1], [1]])
    return out

def transres1fn(x):
    return tf.map_fn(lambda x1: res1fn(x1), x, dtype=tf.float32)


with tf.variable_scope('res1'):
    res1 = tf.map_fn(lambda x: transres1fn(x), elems1, dtype=tf.float32)
print(res1)


with tf.Session() as sess:
    ar1 = np.random.rand(8,5,4,2).astype(np.float32)
    ar2 = np.random.rand(8,5,4,2).astype(np.float32)
    sess.run(tf.global_variables_initializer())
    # print(sess.run(ax3))
    # print(ar1)
    # print(ar2)
    # print("\n", sess.run(alternate, feed_dict={a: ar1, b: ar2}))
    # a1 = sess.run(alternate, feed_dict={a: ar1, b: ar2})
    # print(tf.rank(a1))
    # print(a1.eval())
    a2 = sess.run(d)
    sess.run(res1, {a1: ar1, b1: ar2})
    # print(a2)
