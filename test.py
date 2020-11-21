import tensorflow as tf
import numpy as np

with tf.Session().as_default() as sess:
    input = tf.placeholder(tf.float64, shape=[None, 3, 3])
    print(input.shape)
    op = tf.layers.flatten(input)
    print(op.shape)
    op = op[3]
    print(op.shape)
    sess.run(tf.global_variables_initializer())

    input_data = np.arange(5 * 3 * 3).reshape((5, 3, 3))
    ret = sess.run(op, feed_dict={input: input_data})

    print(ret)
