import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weight = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weight) + biases

    if activation_function is not None:
        outputs = activation_function(Wx_plus_b)
    else:
        outputs = Wx_plus_b

    return outputs

def main():
    x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
    y_data = np.square(x_data) + noise - 0.5

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x_data, y_data)
    plt.ion()

    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

    l1 = add_layer(xs, 1, 10, tf.nn.relu)
    prediction = add_layer(l1, 10, 1)

    loss = tf.reduce_mean(tf.square(ys - prediction), reduction_indices=[0])

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(1000):
            sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
            if i % 50 == 0:
                # to visualize the result and improvement
                try:
                    ax.lines.remove(lines[0])
                except Exception:
                    pass
                prediction_value = sess.run(prediction, feed_dict={xs: x_data})
                # plot the prediction
                lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
                plt.pause(0.1)
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    main()