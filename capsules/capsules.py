import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def inference(images):
    images = tf.cast(images, tf.float32)

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

    with tf.variable_scope("capsule_weights1") as scope:
        capsule_weights1 = tf.Variable(tf.random_normal(shape=[32,10,16,8]), name=scope.name)


    with tf.variable_scope("weight_dot") as scope:
        # elems = (capsule_blocks, capsule_weights1)
        # prediction_vec1_transposed = caps_dot(elems) # shape = [?,32,6,6,10,16]
        prediction_vec1_transposed = tf.map_fn(
            fn = lambda x: caps_dot(x, capsule_weights1),
            elems = capsule_blocks,
            dtype = tf.float32
        )
        prediction_vec1 = tf.transpose(prediction_vec1_transposed, [0,4,1,2,3,5]) # shape = [?,10,32,6,6,16]

    with tf.variable_scope("routing1") as scope:
        vs1 = routing(prediction_vec1, 5, 2)

    with tf.variable_scope("fc1") as scope:
        # vs1 = tf.expand_dims(vs, 0) # Will change later once algorithm optimized to batch size
        vs_flattened = tf.contrib.layers.flatten(vs1)
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

    return vs1, fc_final

def caps(inputs, n_blocks, grid_size, depth, r):
    # shape inputs = [?,32,6,6,8]
    # shape inputs = [BATCH_SIZE, in_n_blocks, in_grid_size, in_depth]
    batch_size = tf.shape(inputs)[0]
    in_n_blocks = tf.shape(inputs)[1]
    in_grid_size = tf.shape(inputs)[2,3]
    in_depth = tf.shape(inputs)[4]
    with tf.variable_scope("capsule_weights1") as scope:
        capsule_weights = tf.Variable(tf.random_normal(shape=[in_n_blocks,n_blocks,grid_size,depth,in_depth]), name=scope.name)
        # shape = [32,10,(1,1),16,8]

    with tf.variable_scope("weight_dot") as scope:
        # prediction_vec1_transposed = caps_dot(elems) # shape = [?,32,6,6,10,(1,1),16]
        prediction_vec_transposed = tf.map_fn(
            fn = lambda x: caps_dot(x, capsule_weights),
            elems = capsule_blocks,
            dtype = tf.float32
        )
        prediction_vec = tf.transpose(prediction_vec_transposed, [0,4,5,6,1,2,3,7])
        # shape = [?,10,1,1,32,6,6,16]

    with tf.variable_scope("routing1") as scope:
        vs1 = _routing(prediction_vec, r) # shape = [?,10,1,1,16]
        # shape = [BATCH_SIZE, out_n_blocks, out_grid_size, out_depth]

def _routing(prediction_vectors, r):
    # shape prediction_vectors = [?,10,1,1,32,6,6,16]
    batch_size = tf.shape(prediction_vectors)[0]
    out_n_blocks = tf.shape(prediction_vectors)[1]
    out_grid_size = tf.shape(prediction_vectors)[2,3]
    in_n_blocks = tf.shape(prediction_vectors)[4]
    in_grid_size = tf.shape(prediction_vectors)[5,6]
    out_depth = tf.shape(prediction_vectors)[7]

    with tf.variable_scope("capsule_layer") as scope:
        b0 = tf.stack([batch_size,out_n_blocks,out_grid_size,in_n_blocks,in_grid_size,1])
        b = tf.fill(b01, 0.0)
        c = tf.nn.softmax(b0, dim=1)
        i = tf.constant(0)
        v0 = tf.stack([batch_size,out_n_blocks,out_grid_size,out_depth])
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
                s = tf.reduce_sum(s_partial, axis=[4,5,6) # shape = [BATCH_SIZE,10,1,1,16]

            with tf.variable_scope("vector_output") as scope:
                v = squash(s1) # shape = [BATCH_SIZE,10,1,1,16]

            with tf.variable_scope("agreement") as scope:
                elems = (prediction_vectors, v) # [BATCH_SIZE,10,1,1,32,6,6,16], [BATCH_SIZE,10,1,1,16]
                agreement = nested_caps_dot(elems,3) # shape = [BATCH_SIZE,10,1,1,32,6,6]
                agreement_expanded = tf.expand_dims(agreement, 6)
                tf.add(b, agreement_expanded)

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

def tensor_dot(tup):
    # shape tup = ([?,6,6,8], [16,10,8])
    with tf.variable_scope("caps_dot") as scope:
        t1 = tup[0]
        t2 = tup[1]
        t1_rank = t1.get_shape().ndims
        t2_rank = t2.get_shape().ndims
        axes = [[t1_rank-1], [t2_rank-1]]
        out = tf.tensordot(t1, t2, axes)
    return out

def caps_dot(t1, t2):
    tup = (t1, t2)
    return tf.map_fn(lambda x: tensor_dot(x), tup, dtype=tf.float32)

def nested_caps_dot(tup, i):
    if i == 0:
        return caps_dot(tup[0], tup[1])
    else:
        return tf.map_fn(lambda x: nested_caps_dot(x, i-1), tup, dtype=tf.float32)

def margin_loss(vs, T):
    # Works as intended
    # shape vs = [?, 10, 16]
    # shape T = [?, 10]
    # shape v_k = [?, 1, 16]
    # shape L = [?,10]
    m_plus = 0.9
    m_minus = 0.1
    lambda_val = 0.5

    with tf.variable_scope('margin_loss') as scope:
        first_max = tf.square(tf.maximum(0., m_plus - tf.norm(vs, axis=2))) # shape = [?,10]
        second_max = tf.square(tf.maximum(0., tf.norm(vs, axis=2) - m_minus)) # shape = [?,10]
        L = T*first_max + lambda_val*(1 - T)*second_max # shape = [?,10]

    return L

def reconstruction_loss(real_im, recon_im):
    with tf.variable_scope("rec_params") as scope:
        real_im_flattened = tf.contrib.layers.flatten(real_im)
        the_square = tf.square(real_im_flattened - recon_im)
        the_sum = tf.reduce_sum(tf.square(real_im_flattened - recon_im), axis=[1])
        rec_loss = tf.reduce_mean(tf.reduce_sum(tf.square(real_im_flattened - recon_im), axis=1))
        return rec_loss

def total_loss(margin_loss, reconstruction_loss):
    #shape margin_loss = [?,10]
    total_margin_loss1 = tf.reduce_sum(margin_loss, axis=1)
    total_margin_loss = tf.reduce_mean(total_margin_loss1)
    total_loss = total_margin_loss + 0.0005 * reconstruction_loss
    return total_loss

if __name__ == "__main__":
    main()
