from keras.models import *
from keras.layers import *
from keras.applications.inception_resnet_v2 import InceptionResNetV2

def xceptionModel(input_shape):
    img_input = Input(shape=input_shape)
    baseModel = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=img_input, pooling="avg")
    x = Dense(1000, activation='relu')(baseModel.output)
    x = Dense(100, activation='softmax', name='output')(x)
    model = Model(img_input, outputs=x)

    return model
