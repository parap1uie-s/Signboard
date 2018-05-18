from keras.models import *
import keras.layers as KL
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
from STN import transformer

def ResNet(input_shape,architecture='resnet50'):
    img_input = Input(shape=input_shape)
    # stage 1
    x = KL.ZeroPadding2D((3, 3))(img_input)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(axis=3, name='bn_conv1')(x)
    x = KL.Activation('relu')(x)
    x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # x = STN_block(x,1)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    # x = STN_block(x,2)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    # x = STN_block(x,3)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i))
    # x = STN_block(x,4)
    # Stage 5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # RNN part
    conv_shape = x.get_shape()
    # print(conv_shape)
    x = KL.Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])))(x)
    x = KL.Dense(1024, activation='relu')(x)
    x = KL.BatchNormalization()(x)
    gru_1 = KL.CuDNNGRU(256, return_sequences=False, name='gru1')(x)
    gru_1b = KL.CuDNNGRU(256, return_sequences=False, go_backwards=True, name='gru1_b')(x)
    # Final
    x = KL.Average()([gru_1, gru_1b])
    x = KL.BatchNormalization()(x)
    x = KL.Dropout(0.25)(x)
    x = KL.Dense(512, activation='relu')(x)
    x = KL.Dropout(0.3)(x)
    x = KL.Dense(100, activation='softmax',name="output")(x)
    model = Model(img_input, outputs=x)
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