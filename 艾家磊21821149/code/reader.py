from os import listdir
from os.path import isfile, join
import tensorflow as tf

def prepose_image(image,height,width):
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height,width],
                                             align_corners=False)
    image = tf.squeeze(image)
    image.set_shape([height, width, 3])
    image = tf.to_float(image)
    means=[123.68,116.78,103.94]
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(image, num_channels, 2)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(channels, 2)

def mean_add(image):
    means=[123.68,116.78,103.94]
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(image, num_channels, 2)
    for i in range(num_channels):
        channels[i] += means[i]
    return tf.concat(channels, 2)

def get_image(path, height, width):
    png = path.lower().endswith('png')
    img_bytes = tf.read_file(path)
    image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)
    return prepose_image(image, height, width)

def get_train_image(batch_size, height, width, path, epochs=2, shuffle=True):
    filenames = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    if not shuffle:
        filenames = sorted(filenames)

    png = filenames[0].lower().endswith('png')  # If first file is a png, assume they all are

    filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle, num_epochs=epochs)
    reader = tf.WholeFileReader()
    _, img_bytes = reader.read(filename_queue)
    image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)

    processed_image = prepose_image(image, height, width)
    return tf.train.batch([processed_image], batch_size, dynamic_pad=True)
    #return image