import keras.backend as K


def mean_error(y_true, y_pred):
    return K.mean(y_pred - y_true)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
