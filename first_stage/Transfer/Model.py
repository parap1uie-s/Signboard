from keras.models import *
import keras.layers as KL
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
from STN import transformer
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet201
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.nasnet import NASNetMobile

def NASTransfer(input_shape, channel=3, final_activation='softmax'):
    input_tensor = KL.Input((input_shape))
    if channel == 3:
        baseModel = NASNetMobile(include_top=False, weights="imagenet", input_tensor=input_tensor, pooling="avg")
    elif channel == 4:
        gray_tensor = RGB2GrayLayer()(input_tensor) # 224,224,1
        x = KL.Concatenate(axis=-1)([input_tensor, gray_tensor]) # 224,224,4
        baseModel = NASNetMobile(include_top=False, weights=None, input_tensor=x, pooling="avg")
    elif channel == 5:
        gray_tensor = RGB2GrayLayer()(input_tensor) # 224,224,1
        sobel_tensor = Gray2SobelEdgeLayer()(gray_tensor) # 224,224,1
        sobel_tensor = KL.Conv2D(64,(14,14), padding="same", name="sobelCONV")(sobel_tensor)
        x = KL.Concatenate(axis=-1)([input_tensor, gray_tensor, sobel_tensor]) # 224,224,4+64
        x = ChannelVoteBlock(x)
        baseModel = NASNetMobile(include_top=False, weights=None, input_tensor=x, pooling="avg")

    x = baseModel.output
    x = KL.Dense(1024, activation='relu')(x)
    x = KL.Dropout(0.3)(x)
    x = KL.Dense(100, activation=final_activation, name='output')(x)
    model = Model(input_tensor, outputs=x)
    return model

def InceptionTransfer(input_shape, channel=3, final_activation='softmax'):
    input_tensor = KL.Input((input_shape))
    if channel == 3:
        baseModel = InceptionV3(include_top=False, weights="imagenet", input_tensor=input_tensor, pooling="avg")
    elif channel == 4:
        gray_tensor = RGB2GrayLayer()(input_tensor) # 224,224,1
        x = KL.Concatenate(axis=-1)([input_tensor, gray_tensor]) # 224,224,4
        baseModel = InceptionV3(include_top=False, weights=None, input_tensor=x, pooling="avg")
    elif channel == 5:
        gray_tensor = RGB2GrayLayer()(input_tensor) # 224,224,1
        sobel_tensor = Gray2SobelEdgeLayer()(gray_tensor) # 224,224,1
        sobel_tensor = KL.Conv2D(64,(14,14), padding="same", name="sobelCONV")(sobel_tensor)
        x = KL.Concatenate(axis=-1)([input_tensor, gray_tensor, sobel_tensor]) # 224,224,4+64
        x = ChannelVoteBlock(x)
        baseModel = InceptionV3(include_top=False, weights=None, input_tensor=x, pooling="avg")

    x = baseModel.output
    x = KL.Dense(1024, activation='relu')(x)
    x = KL.Dropout(0.3)(x)
    x = KL.Dense(100, activation=final_activation, name='output')(x)
    model = Model(input_tensor, outputs=x)
    return model

def Transfer(input_shape, channel=3, final_activation='softmax'):
    input_tensor = KL.Input((input_shape))
    if channel == 3:
        baseModel = InceptionResNetV2(include_top=False, weights="imagenet", input_tensor=input_tensor, pooling="avg")
    elif channel == 4:
        gray_tensor = RGB2GrayLayer()(input_tensor) # 224,224,1
        x = KL.Concatenate(axis=-1)([input_tensor, gray_tensor]) # 224,224,4
        baseModel = InceptionResNetV2(include_top=False, weights=None, input_tensor=x, pooling="avg")
    elif channel == 5:
        gray_tensor = RGB2GrayLayer()(input_tensor) # 224,224,1
        sobel_tensor = Gray2SobelEdgeLayer()(gray_tensor) # 224,224,1
        sobel_tensor = KL.Conv2D(64,(14,14), padding="same", name="sobelCONV")(sobel_tensor)
        x = KL.Concatenate(axis=-1)([input_tensor, gray_tensor, sobel_tensor]) # 224,224,4+64
        x = ChannelVoteBlock(x)
        baseModel = InceptionResNetV2(include_top=False, weights=None, input_tensor=x, pooling="avg")

    x = baseModel.output
    x = KL.Dense(1024, activation='relu')(x)
    x = KL.Dropout(0.3)(x)
    x = KL.Dense(100, activation=final_activation, name='output')(x)
    model = Model(input_tensor, outputs=x)
    return model

def XceptionTransfer(input_shape, channel=3, final_activation='softmax'):
    input_tensor = KL.Input((input_shape))
    if channel == 3:
        baseModel = Xception(include_top=False, weights="imagenet", input_tensor=input_tensor, pooling="avg")
    elif channel == 4:
        gray_tensor = RGB2GrayLayer()(input_tensor) # 224,224,1
        x = KL.Concatenate(axis=-1)([input_tensor, gray_tensor]) # 224,224,4
        baseModel = Xception(include_top=False, weights=None, input_tensor=x, pooling="avg")
    elif channel == 5:
        gray_tensor = RGB2GrayLayer()(input_tensor) # 224,224,1
        sobel_tensor = Gray2SobelEdgeLayer()(gray_tensor) # 224,224,1
        sobel_tensor = KL.Conv2D(64,(14,14), padding="same", name="sobelCONV")(sobel_tensor)
        x = KL.Concatenate(axis=-1)([input_tensor, gray_tensor, sobel_tensor]) # 224,224,4+64
        x = ChannelVoteBlock(x)
        baseModel = Xception(include_top=False, weights=None, input_tensor=x, pooling="avg")

    x = baseModel.output
    x = KL.Dense(1024, activation='relu')(x)
    x = KL.Dropout(0.3)(x)
    x = KL.Dense(100, activation=final_activation, name='output')(x)
    model = Model(input_tensor, outputs=x)
    return model

def DenseNetTransfer(input_shape, channel=3, final_activation='softmax'):
    assert final_activation in ['softmax', 'linear']
    input_tensor = KL.Input((input_shape))
    if channel == 3:
        baseModel = DenseNet201(include_top=False, weights="imagenet", input_tensor=input_tensor, pooling="avg")
    elif channel == 4:
        gray_tensor = RGB2GrayLayer()(input_tensor) # 224,224,1
        x = KL.Concatenate(axis=-1)([input_tensor, gray_tensor]) # 224,224,4
        baseModel = DenseNet201(include_top=False, weights=None, input_tensor=x, pooling="avg")
    elif channel == 5:
        gray_tensor = RGB2GrayLayer()(input_tensor) # 224,224,1
        sobel_tensor = Gray2SobelEdgeLayer()(gray_tensor) # 224,224,1
        sobel_tensor = KL.Conv2D(64,(14,14), padding="same", name="sobelCONV")(sobel_tensor)
        x = KL.Concatenate(axis=-1)([input_tensor, gray_tensor, sobel_tensor]) # 224,224,4+64
        x = ChannelVoteBlock(x)
        baseModel = DenseNet201(include_top=False, weights=None, input_tensor=x, pooling="avg")

    x = baseModel.output
    x = KL.Dense(1024, activation='relu')(x)
    x = KL.Dropout(0.3)(x)
    x = KL.Dense(100, activation=final_activation, name='output')(x)
    model = Model(input_tensor, outputs=x)
    return model

def ResNet(input_shape,architecture='resnet50', channel=3):
    input_tensor = KL.Input((input_shape))
    if channel == 3:
        x = input_tensor
    elif channel == 4:
        gray_tensor = RGB2GrayLayer()(input_tensor) # 224,224,1
        x = KL.Concatenate(axis=-1)([input_tensor, gray_tensor]) # 224,224,4

    # stage 1
    x = KL.ZeroPadding2D((3, 3))(x)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(axis=3, name='bn_conv1')(x)
    x = KL.Activation('relu')(x)
    x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = STN_block(x,1)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    x = STN_block(x,2)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    x = STN_block(x,3)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i))
    x = STN_block(x,4)
    # Stage 5
    x = conv_block(x, 3, [256, 256, 256], stage=5, block='a', strides=(1, 1))
    x = identity_block(x, 3, [256, 256, 256], stage=5, block='b')
    x = identity_block(x, 3, [256, 256, 256], stage=5, block='c')
    x = STN_block(x,5)

    # Final
    x = KL.GlobalAveragePooling2D()(x)
    x = KL.Dense(1024, activation='relu')(x)
    x = KL.Dropout(0.3)(x)
    x = KL.Dense(100, activation=final_activation, name='output')(x)
    model = Model(input_tensor, outputs=x)
    return model

def conv_block(input_tensor, kernel_size, filters, stage, block,
           strides=(2, 2), use_bias=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(axis=3, name=bn_name_base + '1')(shortcut)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x

def identity_block(input_tensor, kernel_size, filters, stage, block,
               use_bias=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x

class SpatialTransformLayer(Layer):

    def __init__(self, output_dim, STN_localization, **kwargs):
        self.output_dim = output_dim
        self.STN_localization = STN_localization
        super(SpatialTransformLayer, self).__init__(**kwargs)

    def call(self, x):
        # return tf.contrib.image.transform(x, self.STN_localization, "BILINEAR")
        return transformer(x, self.STN_localization, (self.output_dim[1], self.output_dim[2]))

    def compute_output_shape(self, input_shape):
        return self.output_dim

class BatchNorm(KL.BatchNormalization):
    """Batch Normalization class. Subclasses the Keras BN class and
    hardcodes training=False so the BN layer doesn't update
    during training.

    Batch normalization has a negative effect on training if batches are small
    so we disable it here.
    """

    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=True)

def STN_block(input_tensor, stage):
    x = KL.Conv2D(256, (3,3), activation='relu', name=str(stage)+"_Conv")(input_tensor)
    x = KL.GlobalAveragePooling2D(name=str(stage)+"_GAvg")(x)
    x = KL.Dense(64, activation='relu', name=str(stage)+"_STN_1" )(x)
    x = KL.Dropout(0.3, name=str(stage)+"_DP_1")(x)
    x = KL.Dense(32, activation='relu', name=str(stage)+"_STN_2" )(x)
    x = KL.Dropout(0.3,name=str(stage)+"_DP_2")(x)
    STN_localization = KL.Dense(6, activation='linear', name=str(stage)+"_STN_3", kernel_initializer="zeros", bias_initializer=bias_init )(x)

    return SpatialTransformLayer(K.int_shape(input_tensor),STN_localization)(input_tensor)

def bias_init(shape, dtype=None):
    return K.variable([1,0,0,0,1,0], dtype=dtype)

class RGB2GrayLayer(Layer):

    def __init__(self, **kwargs):
        super(RGB2GrayLayer, self).__init__(**kwargs)

    def call(self, x):
        return tf.image.rgb_to_grayscale(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], 1)

class Gray2SobelEdgeLayer(Layer):
    """docstring for Gray2SobelEdgeLayer"""
    def __init__(self, **kwargs):
        super(Gray2SobelEdgeLayer, self).__init__(**kwargs)

    def call(self, grayImg):
        w = tf.image.sobel_edges(grayImg)
        res = tf.sqrt( tf.pow(w[...,0,0], 2) + tf.pow(w[...,0,1], 2) )

        # min_by_batch = tf.expand_dims(tf.expand_dims(tf.reduce_min(tf.reduce_min(res, axis=1),axis=1), axis=-1),axis=-1)
        # max_by_batch = tf.expand_dims(tf.expand_dims(tf.reduce_max(tf.reduce_max(res, axis=1),axis=1), axis=-1),axis=-1)

        # res = (res - min_by_batch) / (max_by_batch - min_by_batch)
        res = tf.expand_dims(res, -1)
        res = tf.sigmoid(res)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], 1)

def ChannelVoteBlock(input_tensor):
    filters = int(input_tensor.get_shape()[-1])
    se_shape = (1, 1, filters)
    x = KL.GlobalAveragePooling2D()(input_tensor)
    x = KL.Reshape(se_shape)(x)
    x = KL.Dense(filters, activation='relu',kernel_initializer='he_normal',use_bias=False, name='channel_vote_dense1')(x)
    x = KL.Dense(filters, activation='sigmoid',kernel_initializer='he_normal',use_bias=False, name='channel_vote_dense2')(x)
    x = KL.Multiply()([input_tensor, x])
    return x