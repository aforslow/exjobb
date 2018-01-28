import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

M_PLUS = 0.9
M_MINUS = 0.1
LAMBDA = 0.5


def inference(images):
    images = tf.cast(images, tf.float32)
    batch_size = tf.shape(images)[0]

    if (images.get_shape().ndims < 4):
        images = tf.reshape(images, [batch_size, 28, 28, 1])

    with tf.variable_scope('conv1') as scope:
        conv1 = tf.contrib.layers.conv2d(
            inputs=images,
            num_outputs=256,
            kernel_size=[9,9],
            stride=1,
            padding="VALID",
            activation_fn=tf.nn.relu
        )

    with tf.variable_scope('PrimaryCapsules') as scope:
        primaryCaps = tf.contrib.layers.conv2d(
            inputs=conv1,
            num_outputs=256,
            kernel_size=[9,9],
            stride=2,
            padding="VALID",
            activation_fn=tf.nn.relu
        )
        split_caps = tf.split(
            value=primaryCaps,
            num_or_size_splits=32,
            axis=3,
            name='capsule_grid'
        )
        capsule_blocks = tf.stack(split_caps, 1) # shape = [?,32,6,6,8]
        print(capsule_blocks)

        vs = caps(
            inputs = capsule_blocks,
            n_blocks = 10,
            grid_size = [1,1],
            depth = 16,
            r = 5
            )
        print(vs)
        vs_after_squeeze = tf.squeeze(vs, axis=[2,3])
        print("vs after squeeze:", vs_after_squeeze)

    with tf.variable_scope("fc1") as scope:
        # vs1 = tf.expand_dims(vs, 0) # Will change later once algorithm optimized to batch size
        vs_flattened = tf.contrib.layers.flatten(vs_after_squeeze)
        fc1 = tf.contrib.layers.fully_connected(
        inputs = vs_flattened,
        num_outputs = 512,
        activation_fn = tf.nn.relu
        )

    with tf.variable_scope("fc2") as scope:
        fc2 = tf.contrib.layers.fully_connected(
            inputs = fc1,
            num_outputs = 1024,
            activation_fn = tf.nn.relu
        )

    with tf.variable_scope("fc_final") as scope:
        fc_final = tf.contrib.layers.fully_connected(
            inputs = fc2,
            num_outputs = 784,
            activation_fn = tf.nn.sigmoid
        )

    return vs_after_squeeze, fc_final

def caps(inputs, n_blocks, grid_size, depth, r):
    # shape inputs = [?,32,6,6,8]
    # shape inputs = [BATCH_SIZE, in_n_blocks, in_grid_size, in_depth]
    input_shape = inputs.get_shape().as_list()
    batch_size = tf.shape(inputs)[0]
    in_n_blocks = input_shape[1]
    in_grid_size = input_shape[2:4]
    in_depth = input_shape[4]

    with tf.variable_scope("capsule_weights1") as scope:
        # shape = [32,10,(1,1),16,8]
        capsule_weights = tf.Variable(tf.random_normal(shape=[in_n_blocks,n_blocks,*grid_size,depth,in_depth]),
        validate_shape=True,
        name=scope.name)

    with tf.variable_scope("weight_dot") as scope:
        # shape = [?,32,6,6,10,(1,1),16]
        prediction_vec_transposed = tf.map_fn(
            fn = lambda x: _caps_dot(x, capsule_weights),
            elems = inputs,
            dtype = tf.float32
        )
        prediction_vec = tf.transpose(prediction_vec_transposed, [0,4,5,6,1,2,3,7]) # shape = [?,10,1,1,32,6,6,16]

    with tf.variable_scope("routing1") as scope:
        # shape = [BATCH_SIZE, out_n_blocks, out_grid_size, out_depth]
        vs1 = _routing(prediction_vec, r) # shape = [?,10,1,1,16]
    return vs1

def _routing(prediction_vectors, r):
    # shape prediction_vectors = [?,10,1,1,32,6,6,16]
    pred_vecs_shape = prediction_vectors.get_shape().as_list()
    batch_size = tf.shape(prediction_vectors)[0]
    out_n_blocks = pred_vecs_shape[1]
    out_grid_size = pred_vecs_shape[2:4]
    in_n_blocks = pred_vecs_shape[4]
    in_grid_size = pred_vecs_shape[5:7]
    out_depth = pred_vecs_shape[7]

    with tf.variable_scope("capsule_layer") as scope:
        b0 = tf.stack([batch_size,out_n_blocks,*out_grid_size,in_n_blocks,*in_grid_size,1])
        b = tf.fill(b0, 0.0)
        c = tf.nn.softmax(b, dim=1)
        i = tf.constant(0)
        v0 = tf.stack([batch_size,out_n_blocks,*out_grid_size,out_depth])
        v = tf.fill(v0, 0.0)

        def condition(i, prediction_vectors, v, b, c, r):
            return tf.less(i,r)

        def body(i, prediction_vectors, v, b, c, r):
            # shape prediction_vectors = [BATCH_SIZE,10,1,1,32,6,6,16]
            # shape v = [BATCH_SIZE,10,1,1,16]
            # shape b = [BATCH_SIZE,10,1,1,32,6,6,1]
            # shape c = [BATCH_SIZE,10,1,1,32,6,6,1]
            with tf.variable_scope("coupling"):
                c = tf.nn.softmax(b, dim=1)

            with tf.variable_scope("total_input") as scope:
                s_partial = tf.multiply(c, prediction_vectors) # shape = [BATCH_SIZE,10,1,1,32,6,6,16]
                s = tf.reduce_sum(s_partial, axis=[4,5,6]) # shape = [BATCH_SIZE,10,1,1,16]

            with tf.variable_scope("vector_output") as scope:
                v = squash(s) # shape = [BATCH_SIZE,10,1,1,16]

            with tf.variable_scope("agreement") as scope:
                elems = (prediction_vectors, v) # [BATCH_SIZE,10,1,1,32,6,6,16], [BATCH_SIZE,10,1,1,16]
                agreement = _nested_caps_dot(elems,3, keep_dims=True) # shape = [BATCH_SIZE,10,1,1,32,6,6]
                # agreement_expanded = tf.expand_dims(agreement, 6)
                tf.add(b, agreement)

            return tf.add(i,1), prediction_vectors, v, b, c, r

        _, _, vs, _, _, _ = tf.while_loop(
            cond = lambda i_, pred_vecs, v_, b_, c_, r_: condition(i_, pred_vecs, v_, b_, c_, r_),
            body = lambda i_, pred_vecs, v_, b_, c_, r_: body(i_, pred_vecs, v_, b_, c_, r_),
            loop_vars = [i, prediction_vectors, v, b, c, r]
        )

    return vs

def squash(s):
    # shape s = [BATCH_SIZE, 10, 1, 1, 16]
    # shape s_norm = [BATCH_SIZE, 10, 1, 1]
    with tf.variable_scope("squash") as scope:
        end_idx = s.get_shape().ndims - 1
        s_norm1 = tf.norm(s, axis=end_idx)
        s_norm = tf.expand_dims(s_norm1, end_idx) # shape = [BATCH_SIZE, 10, 1, 1, 1]
        v = tf.square(s_norm)/(1 + tf.square(s_norm)) * (s / s_norm)
    return v

def _tensor_dot(tup, keep_dims=False):
    """
    Calculates the tensordot between the last dimensions of two tensors.

    Inputs: tup = (t1, t2) - Tuple of the two tensors to multiply
    """
    with tf.variable_scope("caps_dot") as scope:
        t1 = tup[0]
        t2 = tup[1]
        t1_rank = t1.get_shape().ndims
        t2_rank = t2.get_shape().ndims
        axes = [[t1_rank-1], [t2_rank-1]]
        out = tf.tensordot(t1, t2, axes)
        if keep_dims:
            return tf.expand_dims(out, t1_rank + t2_rank - 3)
    return out

def _caps_dot(t1, t2, keep_dims=False):
    tup = (t1, t2)
    return tf.map_fn(lambda x: _tensor_dot(x, keep_dims), tup, dtype=tf.float32)

def _nested_caps_dot(tup, i, keep_dims=False):
    if i == 0:
        return _caps_dot(tup[0], tup[1], keep_dims)
    else:
        return tf.map_fn(lambda x: _nested_caps_dot(x, i-1, keep_dims), tup, dtype=tf.float32)

def margin_loss(vs, T):
    # shape vs = [?, 10, 16]
    # shape T = [?, 10]

    with tf.variable_scope('margin_loss') as scope:
        first_max = tf.square(tf.maximum(0., M_PLUS - tf.norm(vs, axis=2))) # shape = [?,10]
        second_max = tf.square(tf.maximum(0., tf.norm(vs, axis=2) - M_MINUS)) # shape = [?,10]
        L = T*first_max + LAMBDA*(1 - T)*second_max # shape = [?,10]

    return L

def reconstruction_loss(real_im, recon_im):
    with tf.variable_scope("rec_params") as scope:
        real_im_flattened = tf.contrib.layers.flatten(real_im)
        the_square = tf.square(real_im_flattened - recon_im)
        the_sum = tf.reduce_sum(tf.square(real_im_flattened - recon_im), axis=[1])
        rec_loss = tf.reduce_mean(tf.reduce_sum(tf.square(real_im_flattened - recon_im), axis=1))
        return rec_loss

def total_loss(margin_logits, reconstruction_logits, images, labels_onehot):
    #shape margin_loss = [?,10]
    batch_margin_loss = margin_loss(margin_logits, labels_onehot)
    reconstruction_loss_ = reconstruction_loss(images, reconstruction_logits)
    total_margin_loss = tf.reduce_mean(tf.reduce_sum(batch_margin_loss, axis=1))
    total_loss = total_margin_loss + 0.0005 * reconstruction_loss_
    return total_loss

def accuracy(caps_logits, labels_onehot):
    capsule_probabilities = tf.norm(caps_logits, axis=2)
    predictions = tf.argmax(capsule_probabilities, axis=1)
    labels = tf.argmax(labels_onehot, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))
    return predictions, labels, accuracy

def train(loss, global_step):
    lr = tf.train.exponential_decay(0.001,
                                  global_step,
                                  100,
                                  0.01,
                                  staircase=True)
    opt = tf.train.AdamOptimizer(lr)
    return opt.minimize(loss)

# inputs1 = tf.placeholder(dtype=tf.float32, shape=[None,784])
# inputs = tf.reshape(inputs1, [tf.shape(inputs1)[0], 28, 28, 1])
# vs, fcs_final = inference(inputs)
# print("inf_out:\n", vs)
# capsule_probabilities = tf.norm(vs, axis=2)
# print("caps_probs:\n", capsule_probabilities)
# predictions = tf.argmax(capsule_probabilities, axis=1)
#
# labels_onehot = tf.placeholder(dtype=tf.float32, shape=[None,10])
# labels = tf.argmax(labels_onehot, axis=1)
# # test_loss_vs = tf.random_normal([3, 10, 16]) / 2
# # y_onehot = tf.one_hot([1,2,3], 10)
# marg_loss = margin_loss(vs, labels_onehot)
# reconstr_loss = reconstruction_loss(inputs, fcs_final)
# norms = tf.norm(vs, axis=2)
# norms0 = norms[0]
# loss = total_loss(marg_loss, reconstr_loss)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))
# # accuracy = tf.metrics.accuracy(labels, predictions)
#
# opt = tf.train.AdamOptimizer(learning_rate=0.0003).minimize(loss)
#
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#
# def main():
#     with tf.Session() as sess:
#         sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
#         for i in range(10000):
#             batch_xs, batch_ys = mnist.train.next_batch(10)
#             # # sess.run(tf.global_variables_initializer())
#             # rand_array = np.random.rand(7, 28, 28, 1)
#             # print(sess.run(inference_out, feed_dict={test_input: rand_array}))
#             v, preds, acc, marg_loss1, loss_val, norm, _ = sess.run(
#                 [vs,
#                 predictions,
#                 accuracy,
#                 marg_loss,
#                 loss,
#                 norms,
#                 opt],
#                 feed_dict={inputs1: batch_xs, labels_onehot: batch_ys})
#             batch_ys_indexes = tf.argmax(batch_ys, axis=1)
#             # print(sess.run(tf.get_default_graph().get_tensor_by_name("routing1/capsule_layer_2/while/vector_output/squash/mul:0")))
#             print("margin loss:", marg_loss1)
#             # print()
#             print("vs:", norm)
#             print()
#             print("Labels:", batch_ys_indexes.eval())
#             print("Preds:", preds)
#             print("Accuracy:", acc)
#             print("Loss:", loss_val)
# if __name__ == "__main__":
#     main()
