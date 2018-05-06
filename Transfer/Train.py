from Model import xceptionModel
from Utils import DataGen
from keras.optimizers import *
from keras.callbacks import EarlyStopping,ModelCheckpoint
import os

if __name__ == '__main__':
    datapath = "/home/Signboard/datasets"
    shape = 139
    model = xceptionModel((shape,shape,3))
    optimizer = SGD(lr=0.001, clipnorm=5.0, momentum=0.9, decay=1e-5)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['acc'])

    if os.path.exists("w.h5"):
        model.load_weights("w.h5", by_name=True, skip_mismatch=True)

    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=0, mode='auto'), 
    ModelCheckpoint("w.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)]

    model.fit_generator(
        DataGen(datapath, shape, batch_size=4, phase='train'), 
        steps_per_epoch=256, 
        epochs=1000, 
        use_multiprocessing=True,
        max_queue_size=100,
        workers=4,
        validation_data=DataGen(datapath, shape, batch_size=4, phase='val'),
        validation_steps=100,
        callbacks=callbacks)
    model.save_weights("w.h5")
    