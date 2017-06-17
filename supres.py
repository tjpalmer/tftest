from numpy import ndarray


def build_autoencoder(samples: ndarray):
    from keras.backend import set_image_data_format
    from keras.layers import Conv2D, UpSampling2D
    from keras.models import Sequential
    from keras.optimizers import Adam
    from numpy import array
    from scipy.stats import norm
    set_image_data_format('channels_last')
    scale_steps = 1
    scale = 1 << scale_steps
    # Build model.
    model = Sequential()
    # Input and initial filter capture.
    filters = 8
    model.add(Conv2D(
        activation='relu',
        filters=filters,
        input_shape=(None, None, 1),
        kernel_size=(3, 3),
        padding='same'))
    model.add(Conv2D(
        activation='relu', filters=filters, kernel_size=(3, 3), padding='same'))
    # Upsample and more texturing.
    for _ in range(scale_steps):
        model.add(UpSampling2D())
        model.add(Conv2D(
            activation='relu', filters=filters, kernel_size=(3, 3),
            padding='same'))
        model.add(Conv2D(
            activation='relu', filters=filters, kernel_size=(3, 3),
            padding='same'))
    # Finishing touches.
    for _ in range(1):
        model.add(Conv2D(
            activation='relu', filters=filters, kernel_size=(3, 3),
            padding='same'))
    model.add(Conv2D(
        activation='relu', filters=1, kernel_size=(3, 3), padding='same'))
    model.compile(loss='mean_squared_error', optimizer=Adam(decay=1e-6))
    # Train.
    # TODO See train_on_batch or fit_generator.
    batch_size = 50
    batch_count = len(samples) // batch_size
    noise = norm(scale=20)
    for i in range(500):
        start = (i % batch_count) * batch_size
        # Prep batch.
        outputs = samples[start:start+batch_size]
        # TODO Different offsets, noise, rotations, flips, sizes? ...
        inputs = outputs[:, ::scale, ::scale]
        # inputs = inputs + noise.rvs(inputs.shape)
        outputs = outputs.reshape([-1] + list(outputs.shape[1:]) + [1])
        inputs = inputs.reshape([-1] + list(inputs.shape[1:]) + [1])
        # Sometimes see where we are.
        if i % 10 == 0:
            print('{}: {}'.format(i, model.test_on_batch(x=inputs, y=outputs)))
        # Train.
        model.train_on_batch(x=inputs, y=outputs)
    # Save the model.
    from datetime import datetime
    from os import makedirs
    from os.path import join
    now = datetime.now()
    time = now.strftime('%Y%m%d-%H%M%S')
    name = 'supres-{}.h5'.format(time)
    dir_name = join('notes', 'models', time)
    makedirs(dir_name)
    out_name = join(dir_name, name)
    model.save(out_name)
    print('Saved to {}'.format(out_name))
    # Return output.
    inputs = samples[:, ::scale, ::scale]
    inputs = inputs.reshape(list(inputs.shape) + [1])
    outputs = model.predict(x=inputs)
    outputs = outputs.reshape(outputs.shape[:-1])
    print(outputs.shape)
    return outputs

def main():
    from argparse import ArgumentParser
    from datetime import datetime
    from matplotlib.pyplot import figure, hist, imshow, show
    from numpy import array, prod
    from scipy.misc import imread
    from scipy.stats import truncnorm
    begin_time = datetime.now()
    print(begin_time)
    parser = ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--random', action='store_true')
    args = parser.parse_args()
    image = imread('450px-Amethyst_gem_stone_texture_wwarby_flickr.jpg')
    image = image.mean(axis=-1)
    figure()
    print(image.shape)
    imshow(image)
    if args.model:
        from keras.models import load_model
        model = load_model(args.model)
        # show_image(image[::4, ::4])
        if args.random:
            # sub = randint(255, size=[16, 32])
            dist = truncnorm(0 / 128, 255 / 128, 0, 128)
            sub = dist.rvs([8, 8])
            sub = sub.reshape([1] + list(sub.shape) + [1])
            count = 7
            for i in range(count):
                # figure()
                # imshow(sub.reshape(sub.shape[1:-1]))
                sub = model.predict(sub)
                sub[sub > 255] = 255
                # figure()
                # imshow(sub.reshape(sub.shape[1:-1]))
                if i < count - 1:
                    sub = 0.9 * sub + 0.1 * dist.rvs(size=sub.shape)
                if max(sub.shape) > 512:
                    break
            out = sub
            # shrunker = randint(255, size=array(shrunk.shape) // 2)
            # shrunk = shrunker.repeat(2, axis=0).repeat(2, axis=1)
            # shrunk = 0.6 * shrunk + 0.4 * randint(255, size=shrunk.shape)
        else:
            # pic = imread('notes/100_0695.JPG').mean(axis=-1)
            # shrunk = pic[::16, ::16]
            shrunk = image[::2, ::2]
            figure()
            imshow(shrunk)
            sub = shrunk
            print(sub.shape)
            out = model.predict(sub.reshape([1] + list(sub.shape) + [1]))
        out = out.reshape(out.shape[1:-1])
    else:
        subs = split(image, 32)
        old_grid = subs.shape[:2]
        print(image.shape)
        print(subs.shape)
        samples = subs.reshape([-1] + list(subs.shape[2:]))
        print(samples.shape)
        images = build_autoencoder(samples)
        print(images.min(), images.max())
        # images = images.reshape([10, 25] + list(images.shape[1:]))
        print('{}'.format(images.shape))
        out = merge(images.reshape(old_grid + images.shape[1:]))
        # out = merge(images)
    out[out > 255] = 255
    end_time = datetime.now()
    print(end_time)
    print(end_time - begin_time)
    figure()
    imshow(out)
    print(out.shape)
    # show_image_grid(images)
    show()


def merge(subs: ndarray):
    from numpy import concatenate
    # Merge vertically, then horizontally.
    merged = concatenate(subs.transpose([0, 2, 1, 3]))
    merged = concatenate(merged.transpose([1, 0, 2]), axis=1)
    return merged


def show_image(image: ndarray):
    from matplotlib.pyplot import imshow, show
    imshow(image)
    show()


def show_image_grid(subs: ndarray):
    from matplotlib.pyplot import imshow, show, subplot2grid
    for i in range(subs.shape[0]):
        for j in range(subs.shape[1]):
            subplot2grid(subs.shape[:2], (i, j))
            imshow(subs[i, j])  # , cmap='gray')
    show()


def split(image: ndarray, size):
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
