import tensorflow as tf

def relu(x):
    return tf.nn.relu(x)

def discriminator_loss(loss_func, real, fake):
    loss = 0
    if loss_func.__contains__('bgan'):
        loss = tf.reduce_mean(fake-real)

    if loss_func == 'lsgan':
        loss = tf.reduce_mean(tf.square(fake)) - tf.reduce_mean(tf.squared_difference(real,1.0))

    if loss_func == 'hinge':
        loss = tf.reduce_mean(relu(1.0+fake))+ tf.reduce_mean(relu(1.0 -real))

    if loss_func == 'wgan':
        loss = -tf.reduce_mean(tf.log(real) + tf.log(1-fake))

    return loss

def generator_loss(loss_func, real, fake):
    loss = 0
    if loss_func.__contains__('bgan'):
        loss = tf.reduce_mean(real-fake)

    if loss_func == 'lsgan':
        loss = tf.reduce_mean(tf.squared_difference(fake,1.0))

    if loss_func == 'hinge':
        loss = -tf.reduce_mean(fake)

    if loss_func == 'wgan':
        loss = -tf.reduce_mean(tf.log(fake))
    return loss


def euclidean_distance(a,b):
    # a.shape = N x D
    # b.shape = M x D
    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    return tf.reduce_mean(tf.square(a-b), axis=2)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def dense(x, in_dim, out_dim):
    weights = weight_variable([in_dim, out_dim])
    bias = bias_variable([out_dim])
    out = tf.add(tf.matmul(x, weights), bias)
    return out



