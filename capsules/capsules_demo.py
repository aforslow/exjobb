import tensorflow as tf
import numpy as np

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
        capsule_blocks = tf.stack(split_caps, 0) # shape = [32,?,6,6,8]
        print(capsule_blocks)

    with tf.variable_scope("capsule_weights1") as scope:
        weight_matrix = tf.random_normal(shape=[32,10,16,8], name=scope.name)

    with tf.variable_scope("weight_dot") as scope:
        elems = (capsule_blocks, weight_matrix)
        prediction_vec1_transposed = caps_dot(elems) # shape = [32,?,6,6,10,16]
        prediction_vec1 = tf.transpose(prediction_vec1_transposed, [1,4,0,2,3,5]) # shape = [?,10,32,6,6,16]
        # temp=[]
        # for i in range(32):
        #     temp.append(tf.tensordot(capsule_blocks[:,i,:,:,:],weight_matrix[i,:,:,:], [[3], [0]]))
        # prediction_vec1 = tf.stack(temp, 1)
        print("preds:", prediction_vec1)
        # prediction_vec1 = capsule_weight_prod(split_caps_stack, weight_matrix, 32) # shape = [BATCH_SIZE,32,6,6,16,10]

    with tf.variable_scope("routing1") as scope:
        vs = routing(prediction_vec1, 3, 2)

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

    return vs, fc_final

def routing(prediction_vectors, r0, l0):
    # shape prediction_vectors = [?,10,32,6,6,16]
    with tf.variable_scope("capsule_layer_%d" % l0) as scope:
        batch_size = tf.shape(prediction_vectors)[0]
        b011 = tf.placeholder(tf.float32, shape=[None,10,32,6,6,1])
        b01 = tf.zeros_like(b011) #tf.zeros(shape=[batch_size,10,32,6,6,1])
        b0 = tf.Variable(b01, name='b0', collections=[tf.GraphKeys.GLOBAL_VARIABLES])
        c0 = tf.nn.softmax(b0)
        i0 = tf.constant(0)
        v01 = tf.placeholder(tf.float32, shape=[None,10,16])
        v0 = tf.zeros_like(v01) #tf.Variable(tf.zeros(shape=[batch_size, 10, 16]), name='v0')

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
    with tf.variable_scope("coupling"):
        c = tf.nn.softmax(b)
        print(c)

    with tf.variable_scope("total_input") as scope:
        s_tensor1 = tf.multiply(c, prediction_vectors) # shape = [BATCH_SIZE,10,32,6,6,16]
        print(s_tensor1)
        print(prediction_vectors)
        s1 = tf.reduce_sum(prediction_vectors, axis=[2,3,4]) # shape = [BATCH_SIZE,10,16]
        print(s1)

    with tf.variable_scope("vector_output") as scope:
        v = squash(s1) # shape = [BATCH_SIZE,10,16]

    with tf.variable_scope("agreement") as scope:
        elems = (prediction_vectors, v)
        agreement1 = nested_caps_dot(elems,1) # shape = [BATCH_SIZE,10,32,6,6]
        tf.add(b, agreement1)

    return tf.add(i,1), prediction_vectors, v, b, c, r

def calculate_agreement(prediction_vectors, v):
    # shape prediction_vectors = [BATCH_SIZE,32,6,6,16,10]
    # shape v = [BATCH_SIZE,16,10]
    # shape out = [BATCH_SIZE,32,6,6,10]
    batch_size = tf.shape(prediction_vectors)[0]
    agreement_for_full_batch_list = []
    for batch in range(10):
        agreement_for_single_batch_list = []
        for j in range(10):
            agreement_for_single_batch_list.append(tf.tensordot(prediction_vectors[batch,:,:,:,:,j], v[batch,:,j], [[3], [0]]))
        # print(prediction_vectors)
        # print(v)
        # print(agreement_for_single_batch_list)
        agreement_single_batch_stack = tf.stack(agreement_for_single_batch_list) # shape = [10, BATCH_SIZE, 32, 6, 6]
        agreement_for_full_batch_list.append(agreement_single_batch_stack)
        print(agreement_for_full_batch_list)
    agreement1_stacked = tf.stack(agreement_for_full_batch_list)
    print(agreement1_stacked)
    agreement1_transposed = tf.transpose(agreement1_stacked, perm=[0, 2, 3, 4, 1]) # shape = [BATCH_SIZE, 32, 6, 6, 10]
    print(agreement1_transposed)
    agreement1_expanded = tf.expand_dims(agreement1_transposed, 4) # shape = [BATCH_SIZE, 32, 6, 6, 1, 10]

    return agreement1_expanded

def squash(s):
    s_norm = tf.norm(s, axis=1)
    print(s_norm)
    print("inside squash")
    print(tf.ones([tf.shape(s)[0]]))
    v = tf.square(s_norm)/(tf.ones([tf.shape(s)[0]]) + tf.square(s_norm)) * (s / s_norm)
    print(v) # shape = [BATCH_SIZE, 16, 10]
    return v

def capsule_weight_prod(t1, t2, n_blocks):
    prediction_vector_list=[]
    for i in range(n_blocks):
        prediction_vector_list.append(tf.tensordot(t1[i,:,:,:],t2[i,:,:,:], [[2], [0]]))
    return tf.stack(prediction_vector_list)

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

def margin_loss(vs, T):
    # shape v = [16, 10]
    # shape T = [1, 10]
    # shape v_k = [16, 1]
    L = []
    m_plus = 0.9
    m_minus = 0.1
    lambda_val = 0.5

    # for k in range(10):
    #     Tk = T[0,k]
    #     first_max = tf.square(tf.maximum(0, m_plus - tf.norm(vs[:,k])))
    #     second_max = tf.square(tf.maximum(0, tf.norm(vs[:,k]) - m_minus))
    #     Lk = Tk*first_max + lambda_val*(1-Tk)*second_max
    #     L.append(Lk)

    first_max = tf.square(tf.maximum(0., m_plus - tf.norm(vs, axis=0))) # shape = [1,10]
    second_max = tf.square(tf.maximum(0., tf.norm(vs, axis=0) - m_minus)) # shape = [1,10]
    L = T*first_max + lambda_val*(tf.ones(tf.shape(T)) - T)*second_max

    return L



test_input = tf.placeholder(dtype=tf.float32, shape=[None,28,28,1])
inference_out = inference(test_input)
test_loss_vs = tf.ones([16, 10]) / 2
y_onehot = tf.one_hot([1,2], 10)
loss = margin_loss(test_loss_vs, y_onehot)

def main():
    with tf.Session() as sess:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        # sess.run(tf.global_variables_initializer())
        rand_array = np.random.rand(10, 28, 28, 1)
        print(sess.run(inference_out, feed_dict={test_input: rand_array}))
        print(sess.run(loss))

if __name__ == "__main__":
    main()
