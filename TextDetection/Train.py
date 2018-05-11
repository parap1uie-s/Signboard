from Model import ResNetBase
from Utils import build_generator
from keras.optimizers import *
from keras.callbacks import EarlyStopping,ModelCheckpoint
import os

if __name__ == '__main__':
    shape = 224
    model = ResNetBase((shape,shape,3))
    optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.005)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['acc'])

    model.load_weights("/home/professorsfx/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True, skip_mismatch=True)

    train_generator = build_generator("dataset/train", augment=True)
    val_generator = build_generator("dataset/val", augment=True)

    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=0, mode='auto'), 
    ModelCheckpoint("text_detection.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)]

    model.fit_generator(
        train_generator,
        steps_per_epoch = len(train_generator),
        epochs=1000, 
        use_multiprocessing=True,
        max_queue_size=100,
        workers=2,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=callbacks)
    model.save_weights("text_detection.h5")
    