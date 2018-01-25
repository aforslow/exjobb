import tensorflow as tf

a = tf.constant([[1,1,1],[2,2,2],[3,4,5]])
b = tf.constant([[1,0,0],[0,1,0],[0,1,1]])
c = tf.tensordot(a,b,[[0], [0]])
d = tf.random_normal([32,6,6,8])
e = tf.random_normal([8,16,32])
f = tf.random_normal([32, 8, 16, 10])

# Test while loop
# shape = [32,6,6,8]
# shape = [32,8,16,10]
# shape out = [32,6,6,16,10]
def test_dot_products(caps_tensor, weight_tensor, caps_dim0, w_dim0):
    i = tf.constant(0)
    stack_tensor = tf.Variable(tf.zeros([32,6,6,16,10]))
    r = tf.while_loop(
        cond = lambda i, capsules, w, stack, caps_dim, w_dim: cond1(i, capsules, w, stack, caps_dim, w_dim),
        body = lambda i, capsules, w, stack, caps_dim, w_dim: body1(i, capsules, w, stack, caps_dim, w_dim),
        loop_vars = [i, caps_tensor, weight_tensor, stack_tensor, caps_dim0, w_dim0]
    )

def body1(i, capsules, w, stack, caps_dim, w_dim):
    dot_prod = tf.tensordot(capsules[i,:,:,:], w[i,:,:,:], [[2], [0]])

    # dot_prod_expanded = tf.expand_dims(dot_prod, 0)
    # stacked_dot_prod = tf.stack([stack, dot_prod_expanded])
    stack[i,:,:,:,:].assign(dot_prod)
    i += 1
    return i, t1, w, stack, caps_dim, w_dim



def cond1(i, capsules, w, stack, caps_dim, w_dim):
    return i < 32

def test_dividing_into_functions():
    b0 = tf.constant([2,2])
    i0 = tf.constant(0)
    r = tf.while_loop(
        cond = lambda i, b: condition(i, b),
        body = lambda i, b: body(i, b),
        loop_vars = [i0, b0]
    )
    return r


def body(i, b):
    with tf.variable_scope('body'):
        a = tf.add(i,1)
        b = tf.add(b, [2,3])
    return a, b

def condition(i, b):
    with tf.variable_scope('cond'):
        a = tf.less(i,3)
    return a


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

def calculate_dot_product(t1, t2, dim1, dim2, split_dim1, split_dim2):
    num_split = tf.shape(t1)[split_dim1]
    t1_list = tf.split(split_dim=split_dim1, num_split=num_split, t1)
    t2_list = tf.split(split_dim=split_dim2, num_split=num_split, t2)
    return tf.tensordot(t1, t2, [[dim1], [dim2]])

def main():
    with tf.Session() as sess:
        print(sess.run(a))
        print(sess.run(b))
        print(sess.run(c))
        # print(sess.run(test_while(d, e)))
        print(sess.run(test_dividing_into_functions()[1]))
        print(sess.run(test_dot_products(d, f, 2, 0)))


if __name__ == "__main__":
    main()
