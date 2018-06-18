from Model import *
from Losses import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--modelType', required=True)
    parser.add_argument('--channel', required=False, default="3")
    parser.add_argument('--loss', required=False, default="cc")
    args = parser.parse_args()

    assert args.modelType in ["densenet", "InceptionResNetV2", "Resnet", "xception", "inception", "nas"]
    assert args.channel.isdigit()
    # categorical_crossentropy or with smooth
    assert args.loss in ['cc', 'ccs', 'focal']
    if args.loss == "cc" or args.loss == "focal":
        activation='softmax'
    else:
        activation='linear'

    args.channel = int(args.channel)

    datapath = "/home/Signboard/second/croped_dataset"
    width = 224
    height = 224
    
    if args.modelType == "densenet":
        model = DenseNetTransfer((height,width,3), channel=args.channel, final_activation=activation)
    elif args.modelType == "InceptionResNetV2":
        model = Transfer((height,width,3), channel=args.channel, final_activation=activation)
    elif args.modelType == "xception":
        model = XceptionTransfer((height,width,3), channel=args.channel, final_activation=activation)
    elif args.modelType == "inception":
        model = InceptionTransfer((height,width,3), channel=args.channel, final_activation=activation)
    elif args.modelType == "nas":
        model = NASTransfer((height,width,3), channel=args.channel, final_activation=activation)

    optimizer = SGD(lr=0.001, clipnorm=5.0, momentum=0.9, decay=1e-5)

    if args.loss == "cc":
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['acc'])
    elif args.loss == "ccs":
        model.compile(optimizer=optimizer, loss=categorical_crossentropy_with_smooth, metrics=['acc'])
    elif args.loss == "focal":
        model.compile(optimizer=optimizer, loss=focal_loss, metrics=['acc'])

    if os.path.exists("Transfer-{}.h5".format(args.modelType)):
        model.load_weights("Transfer-{}.h5".format(args.modelType), by_name=True, skip_mismatch=True)

    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=0, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.01, min_lr=0),
    ModelCheckpoint("Transfer-{}.h5".format(args.modelType), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)]

    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.3,
        rotation_range=30,
        channel_shift_range=10,
        width_shift_range=0.3,
        height_shift_range=0.3,
        # horizontal_flip=True,
        # vertical_flip=True,
        fill_mode="constant",
        cval=0,
        rescale=1.0/255.0)

    val_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = train_datagen.flow_from_directory(
            os.path.join(datapath, "train"),
            target_size=(height, width),
            batch_size=4,
            class_mode='categorical',
            shuffle = True)

    validation_generator = val_datagen.flow_from_directory(
            os.path.join(datapath, "val"),
            target_size=(height, width),
            batch_size=32,
            class_mode='categorical',
            shuffle = True)

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator), 
        epochs=100, 
        use_multiprocessing=True,
        max_queue_size=100,
        workers=4,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=callbacks)
