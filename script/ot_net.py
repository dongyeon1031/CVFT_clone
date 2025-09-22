# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

from VGG import VGG16
# from split_feature import split_feature


def sinkhorn(log_alpha, n_iters=20):
    """Performs incomplete Sinkhorn normalization to log_alpha.
    By a theorem by Sinkhorn and Knopp [1], a sufficiently well-behaved matrix
    with positive entries can be turned into a doubly-stochastic matrix
    (i.e. its rows and columns add up to one) via the succesive row and column
    normalization.
    -To ensure positivity, the effective input to sinkhorn has to be
    exp(log_alpha) (elementwise).
    -However, for stability, sinkhorn works in the log-space. It is only at
    return time that entries are exponentiated.
    [1] Sinkhorn, Richard and Knopp, Paul.
    Concerning nonnegative matrices and doubly stochastic
    matrices. Pacific Journal of Mathematics, 1967
    Args:
    log_alpha: 2D tensor (a matrix of shape [N, N])
    or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
    n_iters: number of sinkhorn iterations (in practice, as little as 20
    iterations are needed to achieve decent convergence for N~100)
    Returns:
    A 3D tensor of close-to-doubly-stochastic matrices (2D tensors are
    converted to 3D tensors with batch_size equals to 1)
    """
    n = log_alpha.get_shape().as_list()[1]
    log_alpha = tf.reshape(log_alpha, [-1, n, n])

    for _ in range(n_iters):
        log_alpha -= tf.reshape(tf.reduce_logsumexp(log_alpha, axis=2), [-1, n, 1])
        log_alpha -= tf.reshape(tf.reduce_logsumexp(log_alpha, axis=1), [-1, 1, n])
    return tf.exp(log_alpha)


def CVFT(x_sat, x_grd, keep_prob, trainable):
    def conv_layer(x, kernel_dim, input_dim, output_dim, stride, trainable, activated,
                   name='ot_conv', activation_function=tf.nn.relu):
        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            initializer = tf.compat.v1.keras.initializers.glorot_uniform()

            weight = tf.get_variable(
                name='weights',
                shape=[kernel_dim, kernel_dim, input_dim, output_dim],
                trainable=trainable,
                initializer=initializer
            )

            bias = tf.get_variable(
                name='biases',
                shape=[output_dim],
                trainable=trainable,
                initializer=tf.zeros_initializer()
            )

            out = tf.nn.conv2d(x, weight, strides=[1, stride, stride, 1], padding='SAME') + bias

            if activated:
                out = activation_function(out)

            return out

    def fc_layer(x, trainable, name='ot_fc'):
        height, width, channel = x.get_shape().as_list()[1:]
        assert channel == 1
        in_dimension = height * width
        out_dimension = in_dimension ** 2

        input_feature = tf.reshape(x, [-1, height * width])

        with tf.compat.v1.variable_scope(name):
            # 가중치 초기화: truncated normal 그대로 유지
            w_init = tf.compat.v1.truncated_normal_initializer(mean=0.0, stddev=0.005)
            # L2 정규화: tf.contrib → keras 정규화로 대체
            l2_reg = tf.compat.v1.keras.regularizers.l2(0.01)

            weight = tf.get_variable(
                name='weights',
                shape=[in_dimension, out_dimension],
                trainable=trainable,
                initializer=w_init,
                regularizer=l2_reg
            )

            # bias 초기화: 기존 코드는 모양이 안 맞을 가능성 큼(eye→flatten 후 길이 in_dimension**2), bih
            # 보통 bias는 0으로 두는 게 안전
            b_init = tf.compat.v1.zeros_initializer()
            bias = tf.get_variable(
                name='biases',
                shape=[out_dimension],
                trainable=trainable,
                initializer=b_init
            )

            out = tf.matmul(input_feature, weight) + bias
            out = tf.reshape(out, [-1, in_dimension, in_dimension])

        return out

    def ot(input_feature, trainable, name='ot'):
        height, width, channel = input_feature.get_shape().as_list()[1:]
        conv_feature = conv_layer(input_feature, kernel_dim=1, input_dim=channel, output_dim=1, stride=1,
                                  trainable=trainable, activated=True, name=name + 'ot_conv')
        fc_feature = fc_layer(conv_feature, trainable, name=name + 'ot_fc')
        ot_matrix = sinkhorn(fc_feature * (-100.))

        return ot_matrix

    def apply_ot(input_feature, ot_matrix):

        height, width, channel = input_feature.get_shape().as_list()[1:]
        in_dimension = ot_matrix.get_shape().as_list()[1]

        reshape_input = tf.transpose(tf.reshape(input_feature, [-1, in_dimension, channel]), [0, 2, 1])
        # shape = [batch, channel, in_dimension]

        out = tf.einsum('bci, bio -> bco', reshape_input, ot_matrix)
        output_feature = tf.reshape(tf.transpose(out, [0, 2, 1]), [-1, height, width, channel])

        return output_feature

    ############## VGG module #################

    vgg_grd = VGG16()
    grd_vgg = vgg_grd.VGG16_conv(x_grd, keep_prob, trainable, 'VGG_grd')
    grd_vgg = conv_layer(grd_vgg, kernel_dim=3, input_dim=512, output_dim=64, stride=2, trainable=trainable,
                         activated=True, name='grd_conv')

    vgg_sat = VGG16()
    sat_vgg = vgg_sat.VGG16_conv(x_sat, keep_prob, trainable, 'VGG_sat')
    sat_vgg = conv_layer(sat_vgg, kernel_dim=3, input_dim=512, output_dim=64, stride=2, trainable=trainable,
                         activated=True, name='sat_conv')

    ############## resize #################
    height, width, channel = sat_vgg.get_shape().as_list()[1:]

    grd_vgg = tf.image.resize_bilinear(grd_vgg, [height, width])

    ############## OT module ######################

    ot_matrix_grd_branch = ot(grd_vgg, trainable, name='ot_grd_branch')
    grd_ot = apply_ot(grd_vgg, ot_matrix_grd_branch)

    sat_ot = sat_vgg

    ################# reshape ###################

    grd_height, grd_width, grd_channel = grd_ot.get_shape().as_list()[1:]
    grd_global = tf.reshape(grd_ot, [-1, grd_height * grd_width * grd_channel])

    sat_height, sat_width, sat_channel = sat_ot.get_shape().as_list()[1:]
    sat_global = tf.reshape(sat_ot, [-1, sat_height * sat_width * sat_channel])

    # return tf.nn.l2_normalize(sat_global, dim=1), tf.nn.l2_normalize(grd_global, dim=1)
    return tf.nn.l2_normalize(sat_global, axis=1), tf.nn.l2_normalize(grd_global, axis=1)


def VGG_conv(x_sat, x_grd, keep_prob, trainable):
    def conv_layer(x, kernel_dim, input_dim, output_dim, stride, trainable, activated,
                   name='ot_conv', activation_function=tf.nn.relu):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):  # reuse=tf.AUTO_REUSE
            weight = tf.get_variable(name='weights', shape=[kernel_dim, kernel_dim, input_dim, output_dim],
                                     trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable(name='biases', shape=[output_dim],
                                   trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())

            out = tf.nn.conv2d(x, weight, strides=[1, stride, stride, 1], padding='SAME') + bias

            if activated:
                out = activation_function(out)

            return out

    ############## VGG module #################

    vgg_grd = VGG16()
    grd_vgg = vgg_grd.VGG16_conv(x_grd, keep_prob, trainable, 'VGG_grd')
    grd_vgg = conv_layer(grd_vgg, kernel_dim=3, input_dim=512, output_dim=64, stride=2, trainable=trainable,
                         activated=True, name='grd_conv')

    vgg_sat = VGG16()
    sat_vgg = vgg_sat.VGG16_conv(x_sat, keep_prob, trainable, 'VGG_sat')
    sat_vgg = conv_layer(sat_vgg, kernel_dim=3, input_dim=512, output_dim=64, stride=2, trainable=trainable,
                             activated=True, name='sat_conv')

    ############## resize #################
    height, width, channel = sat_vgg.get_shape().as_list()[1:]

    grd_vgg = tf.image.resize_bilinear(grd_vgg, [height, width])

    ############## reshape #################
    grd_height, grd_width, grd_channel = grd_vgg.get_shape().as_list()[1:]
    grd_global = tf.reshape(grd_vgg, [-1, grd_height * grd_width * grd_channel])

    sat_height, sat_width, sat_channel = sat_vgg.get_shape().as_list()[1:]
    sat_global = tf.reshape(sat_vgg, [-1, sat_height * sat_width * sat_channel])

    return tf.nn.l2_normalize(sat_global, dim=1), tf.nn.l2_normalize(grd_global, dim=1)


def VGG_gp(x_sat, x_grd, keep_prob, trainable):

    ############## VGG module #################
    vgg_grd = VGG16()
    grd_vgg = vgg_grd.VGG16_conv(x_grd, keep_prob, trainable, 'VGG_grd')

    vgg_sat = VGG16()
    sat_vgg = vgg_sat.VGG16_conv(x_sat, keep_prob, trainable, 'VGG_sat')

    ############## Global pooling #################
    grd_height, grd_width, grd_channel = grd_vgg.get_shape().as_list()[1:]
    grd_global = tf.nn.max_pool(grd_vgg, [1, grd_height, grd_width, 1], [1, 1, 1, 1], padding='VALID')
    grd_global = tf.reshape(grd_global, [-1, grd_channel])

    sat_height, sat_width, sat_channel = sat_vgg.get_shape().as_list()[1:]
    sat_global = tf.nn.max_pool(sat_vgg, [1, sat_height, sat_width, 1], [1, 1, 1, 1], padding='VALID')
    sat_global = tf.reshape(sat_global, [-1, sat_channel])

    return tf.nn.l2_normalize(sat_global, dim=1), tf.nn.l2_normalize(grd_global, dim=1)


def CVFT_ResNet(x_sat, x_grd, keep_prob, trainable):
    from tensorflow.keras.applications import ResNet50
    """
    ResNet50 백본을 써서 VGG_conv과 유사한 파이프라인을 구성:
    - 두 브랜치 모두 ResNet50(conv4 끝) feature 사용
    - 3x3 conv(stride=2)로 채널/해상도 축소
    - grd 해상도를 sat에 맞춰 bilinear resize
    - flatten + L2 normalize
    """
    def undo_vgg_preproc_and_to_resnet_rgb(x):
        # 현재 파이프라인: cv2(BGR) + [103.939,116.779,123.6] 빼둠 → 이를 복원
        mean_bgr = tf.constant([103.939,116.779,123.6], dtype=tf.float32)
        x = x + mean_bgr                      # 원본 BGR 복원 [0..255]
        x = tf.reverse(x, axis=[-1])          # BGR -> RGB
        # Keras ResNet50(Caffe mode) 평균값(RGB 순서) 제거
        mean_rgb = tf.constant([123.68,116.779,103.939], dtype=tf.float32)
        x = x - mean_rgb
        return x

    def resnet_backbone(x, scope):
        x = undo_vgg_preproc_and_to_resnet_rgb(x)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # include_top=False, ImageNet weight 사용
            base = ResNet50(include_top=False, weights='imagenet', input_tensor=x, pooling=None)
            # conv4 끝(feature map 크기/채널이 적당): 'conv4_block6_out'
            feat = base.get_layer('conv4_block6_out').output   # [B, H, W, 1024]
            # 축약 헤드: 3x3 conv, stride=2 → 64채널
            feat = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2,
                                          padding='same', use_bias=True,
                                          name=scope+'_reduce')(feat)
            # (선택) 드롭아웃: keep_prob은 TF1식(keep_prob=0.8 → rate=0.2)
            rate = 1.0 - keep_prob
            feat = tf.keras.layers.Dropout(rate)(feat, training=trainable)
        return feat

    grd_feat = resnet_backbone(x_grd, 'ResNet_grd')
    sat_feat = resnet_backbone(x_sat, 'ResNet_sat')

    # 두 브랜치 공간크기 정합
    h, w, _ = sat_feat.get_shape().as_list()[1:]
    grd_feat = tf.image.resize_bilinear(grd_feat, [h, w])

    # Flatten + L2 normalize
    def flatten_l2(feat):
        shp = feat.get_shape().as_list()[1:]
        vec = tf.reshape(feat, [-1, shp[0]*shp[1]*shp[2]])
        return tf.nn.l2_normalize(vec, axis=1)

    sat_global = flatten_l2(sat_feat)
    grd_global = flatten_l2(grd_feat)
    return sat_global, grd_global