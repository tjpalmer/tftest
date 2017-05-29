def build_autoencoder(shape, samples):
    from hello import bias_variable, conv2d, max_pool_2x2, weight_variable
    from numpy import array
    import tensorflow as tf
    # Inputs.
    batch_size = 50
    shape = array(shape)
    x = tf.placeholder(tf.float32, shape=[None, shape.prod()])
    # y_ = tf.placeholder(tf.float32, shape=[None, shape.prod()])
    # Encode 1.
    W_conv1 = weight_variable([3, 3, 1, 8])
    # b_conv1 = bias_variable([8])
    x_image = tf.reshape(x, [-1] + list(shape) + [1])
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # h_pool1 = max_pool_2x2(h_conv1)
    h_conv1 = tf.nn.relu(tf.nn.conv2d(
        x_image, W_conv1, padding='SAME', strides=[1, 2, 2, 1]))
    h_conv1b = tf.nn.relu(conv2d(h_conv1, weight_variable([3, 3, 8, 8])))
    # Encode 2.
    W_conv2 = weight_variable([3, 3, 8, 8])
    h_conv2 = tf.nn.relu(tf.nn.conv2d(
        h_conv1b, W_conv2, padding='SAME', strides=[1, 2, 2, 1]))
    h_conv2b = tf.nn.relu(conv2d(h_conv2, weight_variable([3, 3, 8, 8])))
    # Encode 3.
    W_conv3 = weight_variable([3, 3, 8, 8])
    h_conv3 = tf.nn.relu(tf.nn.conv2d(
        h_conv2b, W_conv3, padding='SAME', strides=[1, 2, 2, 1]))
    h_conv3b = tf.nn.relu(conv2d(h_conv3, weight_variable([3, 3, 8, 8])))
    # Encode 4.
    h_conv4 = tf.nn.relu(tf.nn.conv2d(
        h_conv3b, weight_variable([3, 3, 8, 8]),
        padding='SAME', strides=[1, 2, 2, 1]))
    h_conv4b = tf.nn.relu(conv2d(h_conv4, weight_variable([3, 3, 8, 8])))
    print('inner shape: {}'.format(h_conv4b.shape))
    # Decode 4.
    h_deconv4 = tf.nn.relu(tf.nn.conv2d_transpose(
        h_conv4b, weight_variable([3, 3, 8, 8]),
        [batch_size] + list(shape//8) + [8], [1, 2, 2, 1]))
    h_deconv4b = tf.nn.relu(conv2d(h_deconv4, weight_variable([3, 3, 8, 8])))
    # h_deconv4c = tf.nn.relu(conv2d(h_deconv4b, weight_variable([3, 3, 8, 8])))
    # h_deconv4d = tf.nn.relu(conv2d(h_deconv4c, weight_variable([3, 3, 8, 8])))
    # h_deconv4e = tf.nn.relu(conv2d(h_deconv4d, weight_variable([3, 3, 8, 8])))
    # h_deconv4f = tf.nn.relu(conv2d(h_deconv4e, weight_variable([3, 3, 8, 8])))
    # Decode 3.
    h_deconv3 = tf.nn.relu(tf.nn.conv2d_transpose(
        h_deconv4b,  weight_variable([3, 3, 8, 8]),
        [batch_size] + list(shape//4) + [8], [1, 2, 2, 1]))
    h_deconv3b = tf.nn.relu(conv2d(h_deconv3, weight_variable([3, 3, 8, 8])))
    h_deconv3c = tf.nn.relu(conv2d(h_deconv3b, weight_variable([3, 3, 8, 8])))
    # h_deconv3d = tf.nn.relu(conv2d(h_deconv3c, weight_variable([3, 3, 8, 8])))
    # h_deconv3e = tf.nn.relu(conv2d(h_deconv3d, weight_variable([3, 3, 8, 8])))
    # h_deconv3f = tf.nn.relu(conv2d(h_deconv3e, weight_variable([3, 3, 8, 8])))
    # Decode 2.
    h_deconv2 = tf.nn.relu(tf.nn.conv2d_transpose(
        h_deconv3c, weight_variable([3, 3, 8, 8]),
        [batch_size] + list(shape//2) + [8], [1, 2, 2, 1]))
    h_deconv2b = tf.nn.relu(conv2d(h_deconv2, weight_variable([3, 3, 8, 8])))
    h_deconv2c = tf.nn.relu(conv2d(h_deconv2b, weight_variable([3, 3, 8, 8])))
    h_deconv2d = tf.nn.relu(conv2d(h_deconv2c, weight_variable([3, 3, 8, 8])))
    h_deconv2e = tf.nn.relu(conv2d(h_deconv2d, weight_variable([3, 3, 8, 8])))
    h_deconv2f = tf.nn.relu(conv2d(h_deconv2e, weight_variable([3, 3, 8, 8])))
    # Decode 1.
    W_deconv1 = weight_variable([3, 3, 8, 8])
    h_deconv1 = tf.nn.relu(tf.nn.conv2d_transpose(
        h_deconv2f, W_deconv1, [batch_size] + list(shape) + [8], [1, 2, 2, 1]))
    h_deconv1b = tf.nn.relu(conv2d(h_deconv1, weight_variable([3, 3, 8, 8])))
    h_deconv1c = tf.nn.relu(conv2d(h_deconv1b, weight_variable([3, 3, 8, 8])))
    h_deconv1d = tf.nn.relu(conv2d(h_deconv1c, weight_variable([3, 3, 8, 8])))
    h_deconv1e = tf.nn.relu(conv2d(h_deconv1d, weight_variable([3, 3, 8, 8])))
    h_deconv1f = tf.nn.relu(conv2d(h_deconv1e, weight_variable([3, 3, 8, 1])))
    y_sample = tf.reshape(h_deconv1f, [-1, shape.prod()])
    # print(h_deconv1f.shape)
    # print(y_sample.shape)
    # return
    # y_conv = tf.reshape(h_deconv1)
    # # Dropout.
    # keep_prob = tf.placeholder(tf.float32)
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # Output.
    # W_fc2 = weight_variable([1024, 10])
    # b_fc2 = bias_variable([10])
    # y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    # Train and test.
    cross_entropy = tf.reduce_mean(tf.square(x - y_sample))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # print('test size: {}'.format(len(mnist.test.labels)))
    batch_count = len(samples) // batch_size
    with sess.as_default():
        # result = h_conv2.eval({x: samples[:1]})
        for i in range(25000):
            start = (i % batch_count) * batch_size
            batch = samples[start:start+batch_size]
            # test_batch = mnist.test.next_batch(50)
            if i % 100 == 0:
                error = cross_entropy.eval({x: batch})
                print('step {}, error {}'.format(i, error))
            #     train_accuracy = accuracy.eval(feed_dict={
            #         x:batch[0], y_: batch[1], keep_prob: 1.0})
            #     print("step %d, training accuracy %g" % (i, train_accuracy))
            #     print("test accuracy %g" % accuracy.eval(feed_dict={
            #         x: test_batch[0],
            #         y_: test_batch[1],
            #         keep_prob: 1.0}))
            train_step.run(feed_dict={x: batch})
        # Save the model.
        from datetime import datetime
        from os import makedirs
        from os.path import join
        now = datetime.now()
        time = now.strftime('%Y%m%d-%H%M%S')
        name = 'texture-{}.cpkt'.format(time)
        saver = tf.train.Saver()
        dir_name = join('notes', 'models', time)
        makedirs(dir_name)
        out_name = saver.save(sess, join(dir_name, name))
        print('Saved to {}'.format(out_name))
        # Render sample output.
        y_image = tf.reshape(y_sample, [-1] + list(shape))
        images = []
        for i in range(batch_count):
            start = i * batch_size
            batch = samples[start:start+batch_size]
            batch_images = y_image.eval({x: batch})
            images.extend(batch_images)
            print('batch images size: {}'.format(batch_images.shape))
        return images


def main():
    from numpy import array, prod
    from scipy.misc import imread
    image = imread('450px-Amethyst_gem_stone_texture_wwarby_flickr.jpg')
    image = image.mean(axis=-1)
    subs = split(image, 32)
    print(image.shape)
    print(subs.shape)
    samples = subs.reshape([prod(subs.shape[:2]), prod(subs.shape[2:])])
    print(samples.shape)
    # show_image(image)
    # show_image_grid(subs)
    images = build_autoencoder(subs.shape[2:], samples)
    images = array(images)
    images = images.reshape([10, 25] + list(images.shape[1:]))
    print('{}'.format(images.shape))
    show_image_grid(images)


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
