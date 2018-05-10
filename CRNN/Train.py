from Model import ResNet
from Utils import DataGen
from keras.optimizers import *
from keras.callbacks import EarlyStopping,ModelCheckpoint
import os

if __name__ == '__main__':
    datapath = "/home/Signboard/datasets"
    width = 224
    height = 75
    model = ResNet((height,width,3))
    optimizer = SGD(lr=0.01, clipnorm=5.0, momentum=0.9, decay=1e-5)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['acc'])

    if os.path.exists("w.h5"):
        model.load_weights("w.h5", by_name=True, skip_mismatch=True)
    else:
        model.load_weights("/home/professorsfx/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True, skip_mismatch=True)

    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=0, mode='auto'), 
    ModelCheckpoint("w.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)]

    model.fit_generator(
        DataGen(datapath, height, width, batch_size=8, phase='train'), 
        steps_per_epoch=256, 
        epochs=1000, 
        use_multiprocessing=True,
        max_queue_size=100,
        workers=4,
        validation_data=DataGen(datapath, height, width, batch_size=16, phase='val'),
        validation_steps=20,
        callbacks=callbacks)
    model.save_weights("w.h5")
    