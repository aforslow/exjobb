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
        print(primaryCaps)
        split_caps = tf.split(
            value=primaryCaps,
            num_or_size_splits=32,
            axis=3,
            name='capsule_grid'
        )
        print(split_caps) # shape = list, 32*[?, 6, 6, 8]
        capsule_blocks = tf.stack(split_caps, 1) # shape = [?,32,6,6,8]
        print(capsule_blocks)

    with tf.variable_scope("capsule_weights1") as scope:
        capsule_weights1 = tf.Variable(tf.random_normal(shape=[32,10,16,8], name=scope.name), name='weights1')


    with tf.variable_scope("weight_dot") as scope:
        # elems = (capsule_blocks, capsule_weights1)
        # prediction_vec1_transposed = caps_dot(elems) # shape = [?,32,6,6,10,16]
        prediction_vec1_transposed = tf.map_fn(
            fn = lambda x: caps_dot(x, capsule_weights1),
            elems = capsule_blocks,
            dtype = tf.float32
        )
        prediction_vec1 = tf.transpose(prediction_vec1_transposed, [0,4,1,2,3,5]) # shape = [?,10,32,6,6,16]
        print("preds:", prediction_vec1)

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

def routing(prediction_vectors, r0, l0):
    # shape prediction_vectors = [?,10,32,6,6,16]
    with tf.variable_scope("capsule_layer_%d" % l0) as scope:
        batch_size = tf.shape(prediction_vectors)[0]
        b01 = tf.stack([batch_size, 10,32,6,6,1])
        b0 = tf.fill(b01, 0.0)
        print(b0)
        c0 = tf.nn.softmax(b0, dim=1)
        i0 = tf.constant(0)
        v01 = tf.stack([batch_size, 10, 16])
        v0 = tf.fill(v01, 0.0)
        print(v0)

        _, _, vs, _, _, _ = tf.while_loop(
            cond = lambda i, pred_vecs, v, b, c, r: condition(i, pred_vecs, v, b, c, r),
            body = lambda i, pred_vecs, v, b, c, r: body(i, pred_vecs, v, b, c, r),
            loop_vars = [i0, prediction_vectors, v0, b0, c0, r0]
        )

    return vs

def condition(i, prediction_vectors, v, b, c, r):
    return tf.less(i,r)

def body(i, prediction_vectors, v, b, c, r):
    # shape prediction_vectors = [BATCH_SIZE,10,32,6,6,16]
    # shape v = [BATCH_SIZE,10,16]
    # shape b = [BATCH_SIZE,10,32,6,6,1]
    # shape c = [BATCH_SIZE,10,32,6,6,1]
    with tf.variable_scope("coupling"):
        c = tf.nn.softmax(b, dim=1)
        print(c)

    with tf.variable_scope("total_input") as scope:
        s_tensor1 = tf.multiply(c, prediction_vectors) # shape = [BATCH_SIZE,10,32,6,6,16]
        print(s_tensor1)
        print(prediction_vectors)
        s1 = tf.reduce_sum(s_tensor1, axis=[2,3,4]) # shape = [BATCH_SIZE,10,16]
        print(s1)

    with tf.variable_scope("vector_output") as scope:
        v = squash(s1) # shape = [BATCH_SIZE,10,16]

    with tf.variable_scope("agreement") as scope:
        elems = (prediction_vectors, v)
        agreement1 = nested_caps_dot(elems,1) # shape = [BATCH_SIZE,10,32,6,6]
        agreement1_expanded = tf.expand_dims(agreement1, 4)
        tf.add(b, agreement1_expanded)

    return tf.add(i,1), prediction_vectors, v, b, c, r

def squash(s):
    # shape s = [BATCH_SIZE, 10, 16]
    # shape s_norm = [BATCH_SIZE, 10]
    with tf.variable_scope("squash") as scope:
        s_last_dim_idx = s.get_shape().ndims - 1
        s_norm1 = tf.norm(s, axis=s_last_dim_idx)
        s_norm = tf.expand_dims(s_norm1, s_last_dim_idx) # shape = [BATCH_SIZE, 10, 1]
        print("inside squash")
        print(s_norm)
        print(tf.ones(tf.shape(s)))
        v = tf.square(s_norm)/(1 + tf.square(s_norm)) * (s / s_norm)
        print(v) # shape = [BATCH_SIZE, 10, 16]
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
        print(out)
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
    L = []
    m_plus = 0.9
    m_minus = 0.1
    lambda_val = 0.5

    with tf.variable_scope('margin_loss') as scope:
        print('margin loss')
        first_max = tf.square(tf.maximum(0., m_plus - tf.norm(vs, axis=2))) # shape = [?,10]
        print(first_max)
        second_max = tf.square(tf.maximum(0., tf.norm(vs, axis=2) - m_minus)) # shape = [?,10]
        print(second_max)
        L = T*first_max + lambda_val*(1 - T)*second_max # shape = [?,10]
        # L = (1-T)*second_max
        print(L)

    # equal = tf.assert_equal(L0, L[0,0])

    return L

def reconstruction_loss(real_im, recon_im):
    with tf.variable_scope("rec_params") as scope:
        print("Reconstruction loss:")
        real_im_flattened = tf.contrib.layers.flatten(real_im)
        print(real_im_flattened)
        print(recon_im)
        the_square = tf.square(real_im_flattened - recon_im)
        print(the_square)
        the_sum = tf.reduce_sum(tf.square(real_im_flattened - recon_im), axis=[1])
        print(the_sum)
        rec_loss = tf.reduce_mean(tf.reduce_sum(tf.square(real_im_flattened - recon_im), axis=1))
        print(rec_loss)
        return rec_loss

def total_loss(margin_loss, reconstruction_loss):
    #shape margin_loss = [?,10]
    total_margin_loss1 = tf.reduce_sum(margin_loss, axis=1)
    print(total_margin_loss1)
    total_margin_loss = tf.reduce_mean(total_margin_loss1)
    total_loss = total_margin_loss + 0.0005 * reconstruction_loss
    return total_loss


inputs1 = tf.placeholder(dtype=tf.float32, shape=[None,784])
inputs = tf.reshape(inputs1, [tf.shape(inputs1)[0], 28, 28, 1])
vs, fcs_final = inference(inputs)
print("inf_out:\n", vs)
capsule_probabilities = tf.norm(vs, axis=2)
print("caps_probs:\n", capsule_probabilities)
predictions = tf.argmax(capsule_probabilities, axis=1)

labels_onehot = tf.placeholder(dtype=tf.float32, shape=[None,10])
labels = tf.argmax(labels_onehot, axis=1)
# test_loss_vs = tf.random_normal([3, 10, 16]) / 2
# y_onehot = tf.one_hot([1,2,3], 10)
marg_loss = margin_loss(vs, labels_onehot)
reconstr_loss = reconstruction_loss(inputs, fcs_final)
norms = tf.norm(vs, axis=2)
norms0 = norms[0]
loss = total_loss(marg_loss, reconstr_loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))
# accuracy = tf.metrics.accuracy(labels, predictions)

opt = tf.train.AdamOptimizer(learning_rate=0.0003).minimize(loss)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def main():
    with tf.Session() as sess:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        for i in range(10000):
            batch_xs, batch_ys = mnist.train.next_batch(10)
            # # sess.run(tf.global_variables_initializer())
            # rand_array = np.random.rand(7, 28, 28, 1)
            # print(sess.run(inference_out, feed_dict={test_input: rand_array}))
            v, preds, acc, marg_loss1, loss_val, norm, _ = sess.run(
                [vs,
                predictions,
                accuracy,
                marg_loss,
                loss,
                norms,
                opt],
                feed_dict={inputs1: batch_xs, labels_onehot: batch_ys})
            batch_ys_indexes = tf.argmax(batch_ys, axis=1)
            # print(sess.run(tf.get_default_graph().get_tensor_by_name("routing1/capsule_layer_2/while/vector_output/squash/mul:0")))
            print("margin loss:", marg_loss1)
            # print()
            print("vs:", norm)
            print()
            print("Labels:", batch_ys_indexes.eval())
            print("Preds:", preds)
            print("Accuracy:", acc)
            print("Loss:", loss_val)
if __name__ == "__main__":
    main()
