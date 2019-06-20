import keras
import numpy as np
from tensorflow.python.keras.callbacks import Callback


class ValidationCallbackKs(keras.callbacks.Callback):

    def __init__(self, model):
        super(keras.callbacks.Callback).__init__()
        test_xe_batches, test_xd_batches, test_y_batches, test_y_batches_prev = model.create_validation_data_with_prev_y_steps(slice_point=1500)

        self.test_xe_batches = test_xe_batches
        self.test_xd_batches = test_xd_batches
        self.test_y_batches = test_y_batches
        self.test_y_batches_prev = test_y_batches_prev

        self.model = model
        self.loss = []
        self.val_loss = []

    def on_train_begin(self, logs=None):
        self.loss = []
        self.val_loss = []

    def on_epoch_begin(self, batch, logs=None):
        return

    def on_batch_end(self, batch, logs=None):
        values = []
        for i in range(len(self.test_xe_batches)):
            # Reshape into batch
            test_xe_batches_inf = np.reshape(self.test_xe_batches[i], newshape=(1, self.model.seq_len_in,
                                                                                self.model.input_feature_amount))
            test_xd_batches_inf = np.reshape(self.test_xd_batches[i], newshape=(1, self.model.seq_len_in,
                                                                                self.model.output_feature_amount))
            test_y_batches_inf = np.reshape(self.test_y_batches[i], newshape=(1, self.model.seq_len_out,
                                                                              self.model.output_feature_amount))

            normalized_ys = self.test_y_batches[i]

            prediction = self.model.predict(test_xe_batches_inf, test_xd_batches_inf, test_y_batches_inf,
                                            self.test_y_batches_prev[i], plot=False)

            mse = np.mean(np.square(prediction - normalized_ys))
            values.append(mse)

        self.val_loss.append(np.mean(values))
        return


class ValidationCallbackTf(Callback):

    def __init__(self, model):
        super(Callback).__init__()
        test_xe_batches, test_xd_batches, test_y_batches, test_y_batches_prev = model.create_validation_data_with_prev_y_steps(slice_point=1500)

        self.test_xe_batches = test_xe_batches
        self.test_xd_batches = test_xd_batches
        self.test_y_batches = test_y_batches
        self.test_y_batches_prev = test_y_batches_prev

        self.model = model
        self.loss = []
        self.val_loss = []

    def on_train_begin(self, logs=None):
        self.loss = []
        self.val_loss = []

    def on_epoch_begin(self, batch, logs=None):
        return

    def on_batch_end(self, batch, logs=None):
        values = []
        for i in range(len(self.test_xe_batches)):
            # Reshape into batch
            test_xe_batches_inf = np.reshape(self.test_xe_batches[i], newshape=(1, self.model.seq_len_in,
                                                                                self.model.input_feature_amount))
            test_xd_batches_inf = np.reshape(self.test_xd_batches[i], newshape=(1, self.model.seq_len_in,
                                                                                self.model.output_feature_amount))
            test_y_batches_inf = np.reshape(self.test_y_batches[i], newshape=(1, self.model.seq_len_out,
                                                                              self.model.output_feature_amount))

            normalized_ys = self.test_y_batches[i]

            prediction = self.model.predict(test_xe_batches_inf, test_xd_batches_inf, test_y_batches_inf,
                                            self.test_y_batches_prev[i], plot=False)

            mse = np.mean(np.square(prediction - normalized_ys))
            values.append(mse)

        self.val_loss.append(np.mean(values))
        return