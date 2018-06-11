import tensorflow as tf

def categorical_crossentropy_with_smooth(target, output):
    return tf.losses.softmax_cross_entropy(onehot_labels=target, logits=output, label_smoothing=0.1)