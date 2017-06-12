def build_autoencoder(model_name, shape, samples):
    from hello import bias_variable, conv2d, max_pool_2x2, weight_variable
    from numpy import array
    from scipy.stats import norm
    import tensorflow as tf
    # Inputs.
    batch_size = 50
    shape = array(shape)
    x = tf.placeholder(tf.float32, shape=[None, shape.prod()])
    y_ = tf.placeholder(tf.float32, shape=[None, shape.prod()])
    # Encode 0.
    x_image = tf.reshape(x, [-1] + list(shape) + [1])
    h_conv0 = tf.nn.relu(conv2d(x_image, weight_variable([3, 3, 1, 8])))
    # Encode 1.
    # b_conv1 = bias_variable([8])
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # h_pool1 = max_pool_2x2(h_conv1)
    h_conv1 = tf.nn.relu(tf.nn.conv2d(
        h_conv0, weight_variable([3, 3, 8, 8]),
        padding='SAME', strides=[1, 2, 2, 1]))
    h_conv1b = tf.nn.relu(conv2d(h_conv1, weight_variable([3, 3, 8, 8])))
    # h_conv1c = tf.nn.relu(conv2d(h_conv1b, weight_variable([3, 3, 8, 8])))
    # Encode 2.
    h_conv2 = tf.nn.relu(tf.nn.conv2d(
        h_conv1b, weight_variable([3, 3, 8, 8]),
        padding='SAME', strides=[1, 2, 2, 1]))
    h_conv2b = tf.nn.relu(conv2d(h_conv2, weight_variable([3, 3, 8, 8])))
    print('inner shape: {}'.format(h_conv2b.shape))
    # Decode 2.
    h_deconv2 = tf.nn.relu(tf.nn.conv2d_transpose(
        h_conv2b, weight_variable([3, 3, 8, 8]),
        [batch_size] + list(shape//2) + [8], [1, 2, 2, 1]))
    h_deconv2b = tf.nn.relu(conv2d(h_deconv2, weight_variable([3, 3, 8, 8])))
    # Decode 1.
    W_deconv1 = weight_variable([3, 3, 8, 8])
    h_deconv1 = tf.nn.relu(tf.nn.conv2d_transpose(
        h_deconv2b, W_deconv1, [batch_size] + list(shape) + [8], [1, 2, 2, 1]))
    h_deconv1b = tf.nn.relu(conv2d(h_deconv1, weight_variable([3, 3, 8, 8])))
    h_deconv1c = tf.nn.relu(conv2d(h_deconv1b, weight_variable([3, 3, 8, 1])))
    y_sample = tf.reshape(h_deconv1c, [-1, shape.prod()])
    # # Dropout.
    # keep_prob = tf.placeholder(tf.float32)
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # Output.
    # W_fc2 = weight_variable([1024, 10])
    # b_fc2 = bias_variable([10])
    # y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y_image = tf.reshape(y_sample, [-1] + list(shape))
    # Train and test.
    cross_entropy = tf.reduce_mean(tf.square(y_ - y_sample))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # print('test size: {}'.format(len(mnist.test.labels)))
    batch_count = len(samples) // batch_size
    saver = tf.train.Saver()
    noise = norm(scale=100)
    with sess.as_default():
        if model_name:
            saver.restore(sess, model_name)
            print('Model loaded from {}'.format(model_name))
        else:
            # result = h_conv2.eval({x: samples[:1]})
            for i in range(5000):
                start = (i % batch_count) * batch_size
                batch = samples[start:start+batch_size].copy()
                batch_noise = batch + noise.rvs(batch.shape)
                # test_batch = mnist.test.next_batch(50)
                if i % 100 == 0:
                    error = cross_entropy.eval({x: batch_noise, y_: batch})
                    print('step {}, error {}'.format(i, error))
                #     train_accuracy = accuracy.eval(feed_dict={
                #         x:batch[0], y_: batch[1], keep_prob: 1.0})
                #     print(
                #         "step %d, training accuracy %g" % (i, train_accuracy))
                #     print("test accuracy %g" % accuracy.eval(feed_dict={
                #         x: test_batch[0],
                #         y_: test_batch[1],
                #         keep_prob: 1.0}))
                train_step.run(feed_dict={x: batch_noise, y_: batch})
            # Save the model.
            from datetime import datetime
            from os import makedirs
            from os.path import join
            now = datetime.now()
            time = now.strftime('%Y%m%d-%H%M%S')
            name = 'texture-{}.cpkt'.format(time)
            dir_name = join('notes', 'models', time)
            makedirs(dir_name)
            out_name = saver.save(sess, join(dir_name, name))
            print('Saved to {}'.format(out_name))
        # This works here but not in the separate render function.
        # Something about sessions or such???
        images = []
        for i in range(batch_count):
            start = i * batch_size
            batch = samples[start:start+batch_size]
            batch_images = y_image.eval({x: batch})
            images.extend(batch_images)
            print('batch images size: {}'.format(batch_images.shape))
        return images
    # return batch_size, x, y_image


def main():
    from argparse import ArgumentParser
    from matplotlib.pyplot import hist, show
    from numpy import array, prod
    from scipy.misc import imread
    from scipy.stats import truncnorm
    parser = ArgumentParser()
    parser.add_argument('--model')
    args = parser.parse_args()
    image = imread('450px-Amethyst_gem_stone_texture_wwarby_flickr.jpg')
    image = image.mean(axis=-1)
    subs = split(image, 32)
    print(image.shape)
    print(subs.shape)
    # dist = norm(-80, 30)
    if args.model:
        dist = truncnorm(0 / 70, 255 / 70, 0, 70)
        subs = dist.rvs(subs.shape)
    # print(subs.min(), subs.max())
    # hist(subs.flat, bins=50)
    # show()
    # return
    samples = subs.reshape([prod(subs.shape[:2]), prod(subs.shape[2:])])
    print(samples.shape)
    # show_image(image)
    # show_image_grid(subs)
    # batch_size, x, y_image =
    images = build_autoencoder(
        args.model, subs.shape[2:], samples)
    # images = render_images(batch_size, samples, x, y_image)
    images = array(images)
    print(images.min(), images.max())
    images = images.reshape([10, 25] + list(images.shape[1:]))
    print('{}'.format(images.shape))
    show_image_grid(images)


def render_images(batch_size, samples, x, y_image):
    import tensorflow as tf
    # Render sample output.
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # print('test size: {}'.format(len(mnist.test.labels)))
    batch_count = len(samples) // batch_size
    with sess.as_default():
        images = []
        for i in range(batch_count):
            start = i * batch_size
            batch = samples[start:start+batch_size]
            batch_images = y_image.eval({x: batch})
            images.extend(batch_images)
            print('batch images size: {}'.format(batch_images.shape))
        return images


def show_image(image):
    from matplotlib.pyplot import imshow, show, subplot2grid
    imshow(image)
    show()


def show_image_grid(subs):
    from matplotlib.pyplot import imshow, show, subplot2grid
    for i in range(subs.shape[0]):
        for j in range(subs.shape[1]):
            subplot2grid(subs.shape[:2], (i, j))
            imshow(subs[i, j])  # , cmap='gray')
    show()


def split(image, size):
    from numpy import array
    rows = []
    for i in range(0, image.shape[0], size):
        row = []
        for j in range(0, image.shape[1], size):
            if i + size <= image.shape[0] and j + size <= image.shape[1]:
                row.append(image[i:i+size, j:j+size])
        if row:
            rows.append(row)
    # print(rows)
    return array(rows)


if __name__ == '__main__':
    main()
