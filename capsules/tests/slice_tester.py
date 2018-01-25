import tensorflow as tf
import numpy as np

input1 = tf.placeholder(dtype=tf.float32, shape=[None,2])
input2 = tf.placeholder(dtype=tf.float32, shape=[None,2])
#
#
# data1 = tf.Variable(input1, validate_shape=False)
# data2 = tf.Variable(input2, validate_shape=False)
# add = tf.add(data1, data2)
#
# row = tf.train.slice_input_producer([data1, data2], num_epochs=1, shuffle=False)
# print(row)

# split1 = tf.split(input1)
# split2 = tf.split(input2)

# unstack1 = tf.unstack(input1)

i0 = tf.constant(0)
out = []
def body(i):
    add = tf.add(i, 1)
    return add

def cond(i):
    return tf.less(i, tf.shape(input1)[0])

unstack1 = tf.while_loop(
    cond = lambda i: cond(i),
    body = lambda i: body(i),
    loop_vars = [i0]
)

# unstack1 = tf.while_loop(
#     cond = lambda i, inarg, o: cond(i, inarg, o),
#     body = lambda i, inarg, o: body(i, inarg, o),
#     loop_vars = [i0, input1, out]
# )


print(unstack1)
ar1 = np.random.rand(4,2)
ar2 = np.random.rand(4,2)

def main1():
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        ar11 = ar1.astype(np.float32)
        ar21 = ar2.astype(np.float32)
        print(ar11.dtype)
        print(ar11.shape)
        print(sess.run(init, feed_dict={input1: ar11, input2: ar21}))
        print(sess.run(row, feed_dict={input1: ar11, input2: ar21}))

def main2():
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print(sess.run(unstack1, feed_dict={input1: ar1, input2: ar2}))


# def myfunction(vector):
#   result = tf.reduce_sum(vector)
#   print_result = tf.Print(result, [result], "myfunction called ")
#   return print_result
#
# MAX_ROWS = 10
#
# # input matrix with 2 columns and unknown number of rows (<MAX_ROWS)
# inputs = tf.placeholder(tf.int32, [None, 2])
# # copy of inputs, will need to have a persistent copy of it because we will
# # be fetching rows in different session.run calls
# data = tf.Variable(inputs, validate_shape=False)
# # input producer that iterates over the rows and pushes them onto Queue
# row = tf.train.slice_input_producer([data], num_epochs=1, shuffle=False)[0]
# myfunction_op = myfunction(row)
#
# # this op will save placeholder values into the variable
# init_op = tf.initialize_all_variables()
#
# # Coordinator is not necessary in this case, but you'll need it if you have
# # more than one Queue in order to close all queues together
# sess = tf.Session()
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
# sess.run([init_op], feed_dict={inputs:[[0, 0], [1, 1], [2, 2]]})
#
# try:
#   for i in range(MAX_ROWS):
#     sess.run([myfunction_op])
# except tf.errors.OutOfRangeError:
#   print('Done iterating')
# finally:
#   # When done, ask other threads to stop.
#   coord.request_stop()

if __name__ == "__main__":
    main2()
