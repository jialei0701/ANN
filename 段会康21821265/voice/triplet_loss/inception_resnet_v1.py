# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim

end_points = {}


def stem(net):
    """
    function: stem

    parameters:
        net:
            type: tensor
            func: input data

    returns:
        net:
            type: tensor
            func: output data
    """

    # 250,120,1
    net = slim.conv2d(net, 32, 3, stride=2, scope='stem_conv1_3X3_32')
    end_points['stem_conv1_3X3_32'] = net
    # 125,60,32
    net = slim.conv2d(net, 32, 3, scope='stem_conv2_3X3_32')
    end_points['stem_conv2_3X3_32'] = net
    # 125,60,32
    net = slim.conv2d(net, 64, 3, scope='stem_conv3_3X3_64')
    end_points['stem_conv3_3X3_64'] = net
    # 125,60,64
    net = slim.avg_pool2d(net, 3, stride=(2, 1), scope='stem_avgpool_3X3')
    end_points['stem_avgpool_3X3_2'] = net
    # 63,60,64
    net = slim.conv2d(net, 80, 1, scope='stem_conv4_1X1_80')
    end_points['stem_conv4_1X1_80'] = net
    # 63,60,80
    net = slim.conv2d(net, 192, 3, scope='stem_conv5_3X3_192')
    end_points['stem_conv5_3X3_192'] = net
    # 63,60,192
    net = slim.conv2d(net, 256, 3, stride=2, scope='stem_conv6_3X3_256')
    end_points['stem_conv6_3X3_256'] = net
    # 32,30,256

    return net


# inception-resnet-A
def inception_resnet_a(net,
                       scale=1.0,
                       activation_fn=tf.nn.relu,
                       scope=None,
                       reuse=None):
    """
    function:
        inception_resnet_a

    parameters:
        net:
        scale:
        activation:
        scope:
        reuse:

    returns:
        net:
            type: tensor
            func: output tensor
    """

    # 32,30,256
    with tf.variable_scope(scope, 'resnet_a', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            # 32,30,256
            tower_conv = slim.conv2d(net, 32, 1, scope='conv1_1x1_32')
            # 32,30,32

        with tf.variable_scope('Branch_1'):
            # 32,30,256
            tower_conv1_0 = slim.conv2d(net, 32, 1, scope='conv1_1x1_32')
            # 32,30,32
            tower_conv1_1 = slim.conv2d(
                tower_conv1_0, 32, 3, scope='conv2_3x3_32')
            # 32,30,32

        with tf.variable_scope('Branch_2'):
            # 32,30,256
            tower_conv2_0 = slim.conv2d(net, 32, 1, scope='conv1_1x1_32')
            # 32,30,32
            tower_conv2_1 = slim.conv2d(
                tower_conv2_0, 32, 3, scope='conv2_3x3_32')
            # 32,30,32
            tower_conv2_2 = slim.conv2d(
                tower_conv2_1, 32, 3, scope='conv3_3x3_32')
            # 32,30,32

        mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
        # 32,30,96
        up = slim.conv2d(
            mixed,
            net.get_shape()[3],
            1,
            normalizer_fn=None,
            activation_fn=None,
            biases_initializer=None,
            scope='concat_conv1_1x1_256')
        # 32,30,256
        # 使用残差网络scale = 0.17
        net += scale * up
        # 32,30,256
        if activation_fn:
            net = activation_fn(net)

    return net


# reduction-a
def reduction_a(net, k, l, m, n):
    """
    function:
        reduction_a

    parameters:
        net:
        k:192
        l:192
        m:256
        n:384

    returns
        net:
            type: tensor
            func: output tensor
    """
    # 32,30,256
    with tf.variable_scope('Branch_0'):
        # 32,30,256
        tower_conv = slim.conv2d(
            net, n, 3, stride=2, padding='SAME', scope='conv1_3x3_384')
        # 16,15,384

    with tf.variable_scope('Branch_1'):
        # 32,30,256
        tower_conv1_0 = slim.conv2d(net, k, 1, scope='conv1_1x1_192')
        # 32,30,192
        tower_conv1_1 = slim.conv2d(tower_conv1_0, l, 3, scope='conv1_3x3_192')
        # 32,30,192
        tower_conv1_2 = slim.conv2d(
            tower_conv1_1, m, 3, stride=2, scope='conv3_3x3_256')
        # 16,15,256

    with tf.variable_scope('Branch_2'):
        # 32,30,256
        tower_pool = slim.avg_pool2d(net, 3, stride=2, scope='avgpool_3x3')
        # 16,15,256

    net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
    # 16,15,896

    return net


# Inception-Renset-B
def inception_resnet_b(net,
                       scale=1.0,
                       activation_fn=tf.nn.relu,
                       scope=None,
                       reuse=None):
    """
    function:
        inception_resnet_b

    parameters:
        net:
        scale:
        activation_fn:
        scope:
        reuse:

    returns:
        net:
    """
    # 16,15,896
    with tf.variable_scope(scope, 'resnet_b', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            # 16,15,896
            tower_conv = slim.conv2d(net, 128, 1, scope='conv1_1x1_128')
            # 16,15,128

        with tf.variable_scope('Branch_1'):
            # 16,15,896
            tower_conv1_0 = slim.conv2d(net, 128, 1, scope='conv1_1x1_128')
            # 16,15,128
            tower_conv1_1 = slim.conv2d(
                tower_conv1_0, 128, [1, 7], scope='conv2_1x7_128')
            # 16,15,128
            tower_conv1_2 = slim.conv2d(
                tower_conv1_1, 128, [7, 1], scope='conv3_7x1_128')
            # 16,15,128

        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        # 16,15,256

        up = slim.conv2d(
            mixed,
            net.get_shape()[3],
            1,
            normalizer_fn=None,
            activation_fn=None,
            biases_initializer=None,
            scope='concat_conv1_1x1_894')
        # 16,15,896
        net += scale * up
        if activation_fn:
            net = activation_fn(net)

    return net


def reduction_b(net):
    """
    function:
        reduction_b

    parameters:
        net:

    returns:
        net:
    """
    # 16,15,896
    with tf.variable_scope('Branch_0'):
        # 16,15,896
        tower_pool = slim.avg_pool2d(net, 3, stride=2, scope='avgpool_3x3')
        # 8,8,896
    with tf.variable_scope('Branch_1'):
        # 16,15,896
        tower_conv = slim.conv2d(net, 256, 1, scope='conv1_1x1_256')
        # 16,15,256
        tower_conv_1 = slim.conv2d(
            tower_conv, 384, 3, stride=2, scope='conv2_3x3_384')
        # 8,8,384
    with tf.variable_scope('Branch_2'):
        # 16,15,896
        tower_conv1 = slim.conv2d(net, 256, 1, scope='conv1_1x1_256')
        # 16,15,256
        tower_conv1_1 = slim.conv2d(
            tower_conv1, 256, 3, stride=2, scope='conv2_3x3_256')
        # 8,8,256
    with tf.variable_scope('Branch_3'):
        # 16,15,896
        tower_conv2 = slim.conv2d(net, 256, 1, scope='conv1_1x1_256')
        # 16,15,256
        tower_conv2_1 = slim.conv2d(tower_conv2, 256, 3, scope='conv2_3x3_256')
        # 16,15,256
        tower_conv2_2 = slim.conv2d(
            tower_conv2_1, 256, 3, stride=2, scope='conv3_3x3_256')
        # 8,8,256

    net = tf.concat([tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool],
                    3)
    # 8*8*1792

    return net


# Inception-Resnet-C
def inception_resnet_c(net,
                       scale=1.0,
                       activation_fn=tf.nn.relu,
                       scope=None,
                       reuse=None):
    """
    function:
        inception_resnet_c

    parameters:
        net:
        scale:
        activation_fn:
        scope:
        reuse:

    returns:
        net:
    """
    # 8,8,1792
    with tf.variable_scope(scope, 'inception_resnet_c', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            # 8,8,1792
            tower_conv = slim.conv2d(net, 192, 1, scope='conv1_1x1_192')
            # 8,8,192
        with tf.variable_scope('Branch_1'):
            # 8,8,1792
            tower_conv1_0 = slim.conv2d(net, 192, 1, scope='conv1_1x1_192')
            # 8,8,192
            tower_conv1_1 = slim.conv2d(
                tower_conv1_0, 192, [1, 3], scope='conv2_1x3_192')
            # 8,8,192
            tower_conv1_2 = slim.conv2d(
                tower_conv1_1, 192, [3, 1], scope='conv3_3x1_192')
            # 8,8,192

        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        # 8,8,384
        up = slim.conv2d(
            mixed,
            net.get_shape()[3],
            1,
            normalizer_fn=None,
            activation_fn=None,
            biases_initializer=None,
            scope='concat_1x1_1792')
        # 8,8,1792
        # scale=0.20
        net += scale * up
        if activation_fn:
            net = activation_fn(net)

    return net


def inception_resnet_v1(inputs,
                        is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None,
                        scope='InceptionResnetV1'):
    """Creates the Inception Resnet V1 model.
    """
    end_points = {}

    with tf.variable_scope(scope, 'InceptionResnetV1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope(
                [slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                    stride=1,
                    padding='SAME'):

                # stem
                net = stem(inputs)
                end_points['stem_out'] = net

                # 5 x Inception-resnet-A
                net = slim.repeat(
                    net,
                    5,
                    inception_resnet_a,
                    scale=0.17,
                    scope="inception_resnet_a")
                end_points['inception_resnet_a_out'] = net

                # Reduction-A
                with tf.variable_scope('reduction_a'):
                    net = reduction_a(net, 192, 192, 256, 384)
                    end_points['reduction_a_out'] = net

                # 10 x Inception-Resnet-B
                net = slim.repeat(
                    net,
                    10,
                    inception_resnet_b,
                    scale=0.10,
                    scope="inception_resnet_b")
                end_points['inception_resnet_b_out'] = net

                # Reduction-B
                with tf.variable_scope('reduction_b'):
                    net = reduction_b(net)
                end_points['reduction_b_out'] = net

                # 5 x Inception-Resnet-C
                net = slim.repeat(
                    net,
                    5,
                    inception_resnet_c,
                    scale=0.20,
                    scope="inception_resnet_c")
                end_points['inception_resnet_c_out'] = net

                # Average Pooling层，输出为8×8×1792
                net = slim.avg_pool2d(
                    net,
                    net.get_shape()[1:3],
                    padding='VALID',
                    scope='avgpool_8x8')

                # 扁平除了batch_size维度的其它维度。使输出变为：[batch_size, ...]
                net = slim.flatten(net)

                # dropout层
                net = slim.dropout(
                    net, dropout_keep_prob, is_training=False, scope='Dropout')
                end_points['PreLogitsFlatten'] = net

                # 全链接层。输出为batch_size×128
                net = slim.fully_connected(
                    net,
                    bottleneck_layer_size,
                    activation_fn=None,
                    scope='logits',
                    reuse=False)

    return net


def inception_resnet_v1_mini(inputs,
                             is_training,
                             dropout_keep_prob=0.8,
                             bottleneck_layer_size=128,
                             reuse=None,
                             scope='InceptionResnetV1'):
    """Creates the Inception Resnet V1 model.
    """
    end_points = {}

    with tf.variable_scope(scope, 'InceptionResnetV1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope(
                [slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                    stride=1,
                    padding='SAME'):

                # stem
                net = stem(inputs)
                end_points['stem_out'] = net

                # 1 x Inception-resnet-A
                net = slim.repeat(
                    net,
                    2,
                    inception_resnet_a,
                    scale=0.17,
                    scope="inception_resnet_a")
                end_points['inception_resnet_a_out'] = net

                # Reduction-A
                with tf.variable_scope('reduction_a'):
                    net = reduction_a(net, 192, 192, 256, 384)
                    end_points['reduction_a_out'] = net

                # 2 x Inception-Resnet-B
                net = slim.repeat(
                    net,
                    4,
                    inception_resnet_b,
                    scale=0.10,
                    scope="inception_resnet_b")
                end_points['inception_resnet_b_out'] = net

                # Reduction-B
                with tf.variable_scope('reduction_b'):
                    net = reduction_b(net)
                end_points['reduction_b_out'] = net

                # 1 x Inception-Resnet-C
                net = slim.repeat(
                    net,
                    2,
                    inception_resnet_c,
                    scale=0.20,
                    scope="inception_resnet_c")
                end_points['inception_resnet_c_out'] = net

                # Average Pooling层，输出为8×8×1792
                net = slim.avg_pool2d(
                    net,
                    net.get_shape()[1:3],
                    padding='VALID',
                    scope='avgpool_8x8')

                # 扁平除了batch_size维度的其它维度。使输出变为：[batch_size, ...]
                net = slim.flatten(net)

                # dropout层
                # net = slim.dropout(
                #     net, dropout_keep_prob, is_training=False, scope='Dropout')
                # end_points['PreLogitsFlatten'] = net


                net = slim.fully_connected(
                    net,
                    bottleneck_layer_size,
                    activation_fn=None,
                    scope='logits',
                    reuse=False)

    return net


def inference(images,
              keep_probability,
              phase_train,
              bottleneck_layer_size=128,
              weight_decay=0.0,
              reuse=None):

    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
            weights_initializer=slim.initializers.xavier_initializer(),
            weights_regularizer=slim.l2_regularizer(weight_decay)):
    
        return inception_resnet_v1_mini(
            images,
            is_training=phase_train,
            dropout_keep_prob=keep_probability,
            bottleneck_layer_size=bottleneck_layer_size,
            reuse=reuse)


if __name__ == '__main__':
    net_input = tf.placeholder(
        dtype=tf.float32, shape=[10, 250, 120, 1], name="input")

    net = inference(net_input, 0.8)
    # stem(net_input)
    with tf.Session() as sess:
        var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for i in range(len(var)):
            print(var[i])
