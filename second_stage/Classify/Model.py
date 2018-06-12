from keras.models import *
import keras.layers as KL
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
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
    x = KL.Dense(61, activation=final_activation, name='output')(x)
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
    x = KL.Dense(61, activation=final_activation, name='output')(x)
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
    x = KL.Dense(61, activation=final_activation, name='output')(x)
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
    x = KL.Dense(61, activation=final_activation, name='output')(x)
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
    x = KL.Dense(61, activation=final_activation, name='output')(x)
    model = Model(input_tensor, outputs=x)
    return model

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