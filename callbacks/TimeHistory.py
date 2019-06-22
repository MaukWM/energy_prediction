import time

import keras as ks


class TimeHistoryKs(ks.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class TimeHistoryTf(ks.callbacks.Callback):

    def on_train_batch_begin(self, batch, logs={}):
        return

    def on_train_batch_end(self, batch, logs={}):
        return

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
