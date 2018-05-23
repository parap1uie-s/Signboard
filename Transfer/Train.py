from Model import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import os

if __name__ == '__main__':
    datapath = "/home/Signboard/datasets"
    shape = 224
    modelType = 'InceptionResNetV2'

    if modelType == "densenet":
        model = DenseNetTransfer((shape,shape,3))
    elif modelType == "InceptionResNetV2":
        model = Transfer((shape,shape,3))
    elif modelType == "Resnet":
        model = ResNet((shape,shape,3))
    elif modelType == "xception":
        model = XceptionTransfer((shape,shape,3))
        # model.load_weights("/home/professorsfx/.keras/models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True, skip_mismatch=True)
    
    optimizer = SGD(lr=0.001, clipnorm=5.0, momentum=0.9, decay=1e-5)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['acc'])

    if os.path.exists("Transfer.h5"):
        model.load_weights("Transfer.h5", by_name=True, skip_mismatch=True)
    else:
        model.load_weights("/home/professorsfx/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True, skip_mismatch=True)

    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=0, mode='auto'), 
    ModelCheckpoint("Transfer.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)]

    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.3,
        rotation_range=20,
        # channel_shift_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        # horizontal_flip=True,
        # vertical_flip=True,
        fill_mode="nearest",
        rescale=1.0/255.0)

    val_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = train_datagen.flow_from_directory(
            os.path.join(datapath, "train"),
            target_size=(shape, shape),
            batch_size=4,
            class_mode='sparse',
            shuffle = True)

    validation_generator = val_datagen.flow_from_directory(
            os.path.join(datapath, "val"),
            target_size=(shape, shape),
            batch_size=32,
            class_mode='sparse',
            shuffle = True)
    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator)+1, 
        epochs=100, 
        use_multiprocessing=True,
        max_queue_size=100,
        workers=4,
        validation_data=validation_generator,
        validation_steps=len(validation_generator)+1,
        callbacks=callbacks)
    
