def build_autoencoder(samples):
    from keras.backend import set_image_data_format
    from keras.layers import Conv2D, UpSampling2D
    from keras.models import Sequential
    from keras.optimizers import Adam
    from numpy import array
    set_image_data_format('channels_last')
    scale_steps = 2
    scale = 1 << scale_steps
    # Build model.
    model = Sequential()
    # Input and initial filter capture.
    model.add(Conv2D(
        activation='relu',
        filters=32,
        input_shape=list(array(samples.shape[1:]) >> scale_steps) + [1],
        kernel_size=(3, 3),
        padding='same'))
    model.add(Conv2D(
        activation='relu', filters=32, kernel_size=(3, 3), padding='same'))
    # Upsample and more texturing.
    for _ in range(scale_steps):
        model.add(UpSampling2D())
        model.add(Conv2D(
            activation='relu', filters=32, kernel_size=(3, 3), padding='same'))
        model.add(Conv2D(
            activation='relu', filters=32, kernel_size=(3, 3), padding='same'))
    model.add(Conv2D(
        activation='relu', filters=1, kernel_size=(3, 3), padding='same'))
    model.compile(loss='mean_squared_error', optimizer=Adam())
    # Train.
    # TODO See train_on_batch or fit_generator.
    batch_size = 50
    batch_count = len(samples) // batch_size
    for i in range(200):
        start = (i % batch_count) * batch_size
        # Prep batch.
        outputs = samples[start:start+batch_size]
        inputs = outputs[:, ::scale, ::scale]
        outputs = outputs.reshape([-1] + list(outputs.shape[1:]) + [1])
        inputs = inputs.reshape([-1] + list(inputs.shape[1:]) + [1])
        # Sometimes see where we are.
        if i % 10 == 0:
            print('{}: {}'.format(i, model.test_on_batch(x=inputs, y=outputs)))
        # Train.
        model.train_on_batch(x=inputs, y=outputs)
    # Return output.
    inputs = samples[:batch_size*batch_count, ::scale, ::scale]
    inputs = inputs.reshape(list(inputs.shape) + [1])
    outputs = model.predict(x=inputs)
    outputs = outputs.reshape(outputs.shape[:-1])
    print(outputs.shape)
    return outputs

def main():
    from argparse import ArgumentParser
    from datetime import datetime
    from matplotlib.pyplot import hist, show
    from numpy import array, prod
    from scipy.misc import imread
    from scipy.stats import truncnorm
    begin_time = datetime.now()
    print(begin_time)
    parser = ArgumentParser()
    parser.add_argument('--model')
    args = parser.parse_args()
    image = imread('450px-Amethyst_gem_stone_texture_wwarby_flickr.jpg')
    image = image.mean(axis=-1)
    subs = split(image, 32)
    print(image.shape)
    print(subs.shape)
    samples = subs.reshape([prod(subs.shape[:2])] + list(subs.shape[2:]))
    print(samples.shape)
    images = build_autoencoder(samples)
    print(images.min(), images.max())
    images = images.reshape([10, 25] + list(images.shape[1:]))
    print('{}'.format(images.shape))
    end_time = datetime.now()
    print(end_time)
    print(end_time - begin_time)
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
