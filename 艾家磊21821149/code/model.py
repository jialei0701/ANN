import tensorflow as  tf
from tensorflow.python import pywrap_tensorflow


def relu(input):
    relu = tf.nn.relu(input)
    # convert nan to zero (nan != nan)
    nan_to_zero = tf.where(tf.equal(relu, relu), relu, tf.zeros_like(relu))
    return nan_to_zero

def instance_norm(x):
    epsilon = 1e-9
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))

def conv2d(x, input_filters, output_filters, kernel, strides, mode='REFLECT'):
    with tf.variable_scope('conv'):

        shape = [kernel, kernel, input_filters, output_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        x_padded = tf.pad(x, [[0, 0], [int(kernel / 2), int(kernel / 2)], [int(kernel / 2), int(kernel / 2)], [0, 0]], mode=mode)
        return tf.nn.conv2d(x_padded, weight, strides=[1, strides, strides, 1], padding='VALID', name='conv')

def residual(x, filters, kernel, strides):
    with tf.variable_scope('residual'):
        conv1 = conv2d(x, filters, filters, kernel, strides)
        conv2 = conv2d(relu(conv1), filters, filters, kernel, strides)
        residual = x + conv2
        return residual

def resize_conv2d(x, input_filters, output_filters, kernel, strides, training):
    with tf.variable_scope('conv_transpose'):
        height = x.get_shape()[1].value if training else tf.shape(x)[1]
        width = x.get_shape()[2].value if training else tf.shape(x)[2]

        new_height = height * strides * 2
        new_width = width * strides * 2

        x_resized = tf.image.resize_images(x, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # shape = [kernel, kernel, input_filters, output_filters]
        # weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        return conv2d(x_resized, input_filters, output_filters, kernel, strides)

def load_model(input,model_path):
    net = {}

    reader = pywrap_tensorflow.NewCheckpointReader(model_path)
    vgg_variable = reader.get_variable_to_shape_map()
    keys = sorted(vgg_variable)

    # Print tensor name and values
    # for key in keys:
    #     if key > 'vgg_16/conv6':
    #         print("tensor_name: ", key,reader.get_tensor(key).shape)

    # conv1_1
    net['conv1_1'] = tf.nn.conv2d(input,reader.get_tensor('vgg_16/conv1/conv1_1/weights'),[1,1,1,1],padding='SAME')
    net['conv1_1'] = tf.nn.bias_add(net['conv1_1'],reader.get_tensor('vgg_16/conv1/conv1_1/biases'))
    net['conv1_1'] = tf.nn.relu(net['conv1_1'])

    # conv1_2
    net['conv1_2'] = tf.nn.conv2d(net['conv1_1'],reader.get_tensor('vgg_16/conv1/conv1_2/weights'),[1,1,1,1],padding='SAME')
    net['conv1_2'] = tf.nn.bias_add(net['conv1_2'],reader.get_tensor('vgg_16/conv1/conv1_2/biases'))
    net['conv1_2'] = tf.nn.relu(net['conv1_2'])

    #pool1
    net['pool1'] = tf.nn.max_pool(net['conv1_2'],ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    # conv2_1
    net['conv2_1'] = tf.nn.conv2d(net['pool1'],reader.get_tensor('vgg_16/conv2/conv2_1/weights'),[1,1,1,1],padding='SAME')
    net['conv2_1'] = tf.nn.bias_add(net['conv2_1'],reader.get_tensor('vgg_16/conv2/conv2_1/biases'))
    net['conv2_1'] = tf.nn.relu(net['conv2_1'])

    # conv2_2
    net['conv2_2'] = tf.nn.conv2d(net['conv2_1'],reader.get_tensor('vgg_16/conv2/conv2_2/weights'),[1,1,1,1],padding='SAME')
    net['conv2_2'] = tf.nn.bias_add(net['conv2_2'],reader.get_tensor('vgg_16/conv2/conv2_2/biases'))
    net['conv2_2'] = tf.nn.relu(net['conv2_2'])

    #pool2
    net['pool2'] = tf.nn.max_pool(net['conv2_2'],ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    # conv3_1
    net['conv3_1'] = tf.nn.conv2d(net['pool2'],reader.get_tensor('vgg_16/conv3/conv3_1/weights'),[1,1,1,1],padding='SAME')
    net['conv3_1'] = tf.nn.bias_add(net['conv3_1'],reader.get_tensor('vgg_16/conv3/conv3_1/biases'))
    net['conv3_1'] = tf.nn.relu(net['conv3_1'])

    # conv3_2
    net['conv3_2'] = tf.nn.conv2d(net['conv3_1'],reader.get_tensor('vgg_16/conv3/conv3_2/weights'),[1,1,1,1],padding='SAME')
    net['conv3_2'] = tf.nn.bias_add(net['conv3_2'],reader.get_tensor('vgg_16/conv3/conv3_2/biases'))
    net['conv3_2'] = tf.nn.relu(net['conv3_2'])

    # conv3_3
    net['conv3_3'] = tf.nn.conv2d(net['conv3_2'],reader.get_tensor('vgg_16/conv3/conv3_3/weights'),[1,1,1,1],padding='SAME')
    net['conv3_3'] = tf.nn.bias_add(net['conv3_3'],reader.get_tensor('vgg_16/conv3/conv3_3/biases'))
    net['conv3_3'] = tf.nn.relu(net['conv3_3'])

    #pool3
    net['pool3'] = tf.nn.max_pool(net['conv3_3'],ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    # conv4_1
    net['conv4_1'] = tf.nn.conv2d(net['pool3'],reader.get_tensor('vgg_16/conv4/conv4_1/weights'),[1,1,1,1],padding='SAME')
    net['conv4_1'] = tf.nn.bias_add(net['conv4_1'],reader.get_tensor('vgg_16/conv4/conv4_1/biases'))
    net['conv4_1'] = tf.nn.relu(net['conv4_1'])

    # conv4_2
    net['conv4_2'] = tf.nn.conv2d(net['conv4_1'],reader.get_tensor('vgg_16/conv4/conv4_2/weights'),[1,1,1,1],padding='SAME')
    net['conv4_2'] = tf.nn.bias_add(net['conv4_2'],reader.get_tensor('vgg_16/conv4/conv4_2/biases'))
    net['conv4_2'] = tf.nn.relu(net['conv4_2'])

    # conv4_3
    net['conv4_3'] = tf.nn.conv2d(net['conv4_2'],reader.get_tensor('vgg_16/conv4/conv4_3/weights'),[1,1,1,1],padding='SAME')
    net['conv4_3'] = tf.nn.bias_add(net['conv4_3'],reader.get_tensor('vgg_16/conv4/conv4_3/biases'))
    net['conv4_3'] = tf.nn.relu(net['conv4_3'])

    #pool4
    net['pool4'] = tf.nn.max_pool(net['conv4_3'],ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    # conv5_1
    net['conv5_1'] = tf.nn.conv2d(net['pool4'],reader.get_tensor('vgg_16/conv5/conv5_1/weights'),[1,1,1,1],padding='SAME')
    net['conv5_1'] = tf.nn.bias_add(net['conv5_1'],reader.get_tensor('vgg_16/conv5/conv5_1/biases'))
    net['conv5_1'] = tf.nn.relu(net['conv5_1'])

    # conv5_2
    net['conv5_2'] = tf.nn.conv2d(net['conv5_1'],reader.get_tensor('vgg_16/conv5/conv5_2/weights'),[1,1,1,1],padding='SAME')
    net['conv5_2'] = tf.nn.bias_add(net['conv5_2'],reader.get_tensor('vgg_16/conv5/conv5_2/biases'))
    net['conv5_2'] = tf.nn.relu(net['conv5_2'])

    # conv5_3
    net['conv5_3'] = tf.nn.conv2d(net['conv5_2'],reader.get_tensor('vgg_16/conv5/conv5_3/weights'),[1,1,1,1],padding='SAME')
    net['conv5_3'] = tf.nn.bias_add(net['conv5_3'],reader.get_tensor('vgg_16/conv5/conv5_3/biases'))
    net['conv5_3'] = tf.nn.relu(net['conv5_3'])

    # pool5
    net['pool5'] = tf.nn.max_pool(net['conv5_3'],ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    p5_shape = net['pool5'].shape
    #print(net['pool5'].shape)
    net['reshape'] = tf.reshape(net['pool5'],shape=[-1,p5_shape[1] * p5_shape[2] * p5_shape[3]])

    # # fc6
    # net['fc6'] = tf.matmul(net['reshape'],tf.reshape(reader.get_tensor('vgg_16/fc6/weights'),
    #                                                  shape=[p5_shape[1] * p5_shape[2] * p5_shape[3],4096]))
    # net['fc6'] = tf.add(net['fc6'],reader.get_tensor('vgg_16/fc6/biases'))
    # net['fc6'] = tf.nn.relu(tf.nn.dropout(net['fc6'],keep_prob=0.5))
    #
    # # fc7
    # net['fc7'] = tf.matmul(net['fc6'],tf.reshape(reader.get_tensor('vgg_16/fc7/weights'),
    #                                              shape=[4096,4096]))
    # net['fc7'] = tf.add(net['fc7'],reader.get_tensor('vgg_16/fc7/biases'))
    # net['fc7'] = tf.nn.relu(tf.nn.dropout(net['fc7'],keep_prob=0.5))
    #
    # # fc8
    # net['fc8'] = tf.matmul(net['fc7'],tf.reshape(reader.get_tensor('vgg_16/fc8/weights'),
    #                                              shape=[4096,1000]))
    # net['fc8'] = tf.add(net['fc8'],reader.get_tensor('vgg_16/fc8/biases'))
    # net['fc8'] = tf.nn.relu(net['fc8'])
    #
    # softmax = tf.nn.softmax(net['fc8'])
    #
    # predictions = tf.argmax(softmax, 1)

    return net

def net(image, training=True):

    # Less border effects when padding a little before passing through ..
    image = tf.pad(image, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')

    with tf.variable_scope('conv1'):
        conv1 = relu(instance_norm(conv2d(image, 3, 32, 9, 1)))
    with tf.variable_scope('conv2'):
        conv2 = relu(instance_norm(conv2d(conv1, 32, 64, 3, 2)))
    with tf.variable_scope('conv3'):
        conv3 = relu(instance_norm(conv2d(conv2, 64, 128, 3, 2)))
    with tf.variable_scope('res1'):
        res1 = residual(conv3, 128, 3, 1)
    with tf.variable_scope('res2'):
        res2 = residual(res1, 128, 3, 1)
    with tf.variable_scope('res3'):
        res3 = residual(res2, 128, 3, 1)
    with tf.variable_scope('res4'):
        res4 = residual(res3, 128, 3, 1)
    with tf.variable_scope('res5'):
        res5 = residual(res4, 128, 3, 1)
    # print(res5.get_shape())
    with tf.variable_scope('deconv1'):
        # deconv1 = relu(instance_norm(conv2d_transpose(res5, 128, 64, 3, 2)))
        deconv1 = relu(instance_norm(resize_conv2d(res5, 128, 64, 3, 2, training)))
    with tf.variable_scope('deconv2'):
        # deconv2 = relu(instance_norm(conv2d_transpose(deconv1, 64, 32, 3, 2)))
        deconv2 = relu(instance_norm(resize_conv2d(deconv1, 64, 32, 3, 2, training)))
    with tf.variable_scope('deconv3'):
        # deconv_test = relu(instance_norm(conv2d(deconv2, 32, 32, 2, 1)))
        deconv3 = tf.nn.tanh(instance_norm(conv2d(deconv2, 32, 3, 9, 1)))

    y = (deconv3 + 1) * 127.5
    # Remove border effect reducing padding.
    height = tf.shape(y)[1]
    width = tf.shape(y)[2]
    y = tf.slice(y, [0, 10, 10, 0], tf.stack([-1, height - 20, width - 20, -1]))
    return y