import keras.backend as K


def mean_error(y_true, y_pred):
    return K.mean(y_pred - y_true)