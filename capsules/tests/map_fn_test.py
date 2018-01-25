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
        end1 = tf.rank(t1)-1
        end2 = tf.rank(t2)-1
        axes = tf.expand_dims(tf.stack([end1, end2]), 1)
        out = tf.tensordot(t1, t2, axes)
    return out

def caps_dot(tup):
    return tf.map_fn(lambda x: tensor_dot(x), tup, dtype=tf.float32)

def nested_caps_dot(tup, i):
    if i == 0:
        return caps_dot(tup)
    else:
        return tf.map_fn(lambda x: nested_caps_dot(x, i-1), tup, dtype=tf.float32)

# tf.map(lambda x: tf.map(lambda y: cap))
# elems = (np.array([1, 2, 3], dtype=np.int64), np.array([-1, 1, -1], dtype=np.int64))
# a = tf.placeholder(tf.float32, [None, 3, 2])
# b = tf.placeholder(tf.float32, [None, 4, 2])
a = tf.ones([32,5,6,3,2])
b = tf.ones([32,5,6,4,2])
elems = (a,b)
# alternate = tf.map_fn(lambda x: nested_map(x), elems, dtype=tf.float32)
alternate = nested_caps_dot(elems, 2)
c = tf.ones([32,5,6,3,4])
d = alternate + c
print(d)
# alternate == [-1, 2, -3]

# c = tf.ones([2,3,4])
# d = tf.ones([4,5,2,3])
# e1 = tf.rank(c)-1
# e2 = tf.rank(d)-1
# ax1 = tf.stack([e1, e2])
# ax2 = [e1, e2]
# ax3 = tf.expand_dims(ax1, 1)
# print(ax3)

with tf.Session() as sess:
    ar1 = np.random.rand(5,3,2).astype(np.float32)
    ar2 = np.random.rand(5,4,2).astype(np.float32)
    sess.run(tf.global_variables_initializer())
    # print(sess.run(ax3))
    # print(ar1)
    # print(ar2)
    # print("\n", sess.run(alternate, feed_dict={a: ar1, b: ar2}))
    # a1 = sess.run(alternate, feed_dict={a: ar1, b: ar2})
    # print(tf.rank(a1))
    # print(a1.eval())
    a2 = sess.run(d)
