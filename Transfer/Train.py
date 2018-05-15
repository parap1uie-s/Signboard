from Model import *
from Utils import DataGen
from keras.optimizers import *
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import os

if __name__ == '__main__':
    datapath = "/home/Signboard/datasets"
    shape = 224
    # model = ResNet((shape,shape,3))
    model = XceptionTransfer((shape,shape,3))
    optimizer = SGD(lr=0.001, clipnorm=5.0, momentum=0.9, decay=1e-5)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['acc'])

    if os.path.exists("w.h5"):
        model.load_weights("w.h5", by_name=True, skip_mismatch=True)
    else:
        model.load_weights("/home/professorsfx/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True, skip_mismatch=True)

    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=0, mode='auto'), 
    ModelCheckpoint("w.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)]

    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="constant",
        cval=0,
        rescale=1.0/255.0)

    val_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = train_datagen.flow_from_directory(
            os.path.join(datapath, "train"),
            target_size=(shape, shape),
            batch_size=2,
            class_mode='sparse',
            shuffle = "false")

    validation_generator = val_datagen.flow_from_directory(
            os.path.join(datapath, "val"),
            target_size=(shape, shape),
            batch_size=32,
            class_mode='sparse',
            shuffle = "false")
    model.fit_generator(
        # DataGen(datapath, shape, batch_size=4, phase='train'), 
        train_generator,
        steps_per_epoch=len(train_generator), 
        epochs=100, 
        use_multiprocessing=True,
        max_queue_size=100,
        workers=4,
        # validation_data=DataGen(datapath, shape, batch_size=16, phase='val'),
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=callbacks)
    