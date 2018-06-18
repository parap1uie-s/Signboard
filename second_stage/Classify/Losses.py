import tensorflow as tf
from keras import backend as K

def categorical_crossentropy_with_smooth(target, output):
    return tf.losses.softmax_cross_entropy(onehot_labels=target, logits=output, label_smoothing=0.1)


'''
Compatible with tensorflow backend
'''
def focal_loss(y_true, y_pred):
    alpha=.75
    gamma=2.
    y_pred=K.clip(y_pred,K.epsilon(),1.-K.epsilon())#improve the stability of the focal loss and see issues 1 for more information
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
