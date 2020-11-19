from tensorflow import TensorShape
from tensorflow.python.framework import tensor_shape

tensorShape: TensorShape = tensor_shape.as_shape((None, 1, 2))

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
