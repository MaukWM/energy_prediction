import pickle

from metrics import mean_error

import numpy as np
import matplotlib.pyplot as plt


class Model:

    def __init__(self, name, data_dict, batch_size=256, state_size=42, input_feature_amount=83, output_feature_amount=1,
                 seq_len_in=96, seq_len_out=96, plot_time_steps_view=192, steps_per_epoch=100, epochs=50,
                 learning_rate=0.00075, intermediates=1, agg_level="UNK"):
        self.batch_size = batch_size
        self.state_size = state_size
        self.input_feature_amount = input_feature_amount
        self.output_feature_amount = output_feature_amount
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        self.plot_time_steps_view = plot_time_steps_view
        self.normalized_input_data = data_dict['normalized_input_data']
        self.normalized_output_data = data_dict['normalized_output_data']
        self.output_std = data_dict['output_std']
        self.output_mean = data_dict['output_mean']
        self.name = name
        self.test_train_ratio = 0.5
        self.agg_level = agg_level

        # Generate the validation data
        self.validation_data = self.generate_validation_data()

        # Training info
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.intermediates = intermediates

        # To be determined
        self.model = None
        self.plot_loss = False

        self.validation_metrics = [mean_error]

    def generate_validation_data(self, slice_point=1500):
        """
        Generate validation data of the whole testing set on a slicing point.
        :return: the validation data: [input encoder, input decoder], output_decoder
        """
        test_xe_batches = []
        test_xd_batches = []
        test_y_batches = []

        # Split into testing set
        if hasattr(self.normalized_input_data, 'shape') and hasattr(self.normalized_output_data, 'shape'):
            test_x = self.normalized_input_data[:, -self.normalized_input_data.shape[1] // int((1 / self.test_train_ratio)):]
            test_y = self.normalized_output_data[:, -self.normalized_output_data.shape[1] // int((1 / self.test_train_ratio)):]
        else:
            test_x = []
            test_y = []
            for building in self.normalized_input_data:
                test_x.append(building[-int(np.shape(building)[0] * self.test_train_ratio):])
            for building in self.normalized_output_data:
                test_y.append(building[-int(np.shape(building)[0] * self.test_train_ratio):])

        for i in range(len(test_x)):
            for j in range(len(test_x[i]) - self.seq_len_out - self.seq_len_in):
                #  Change modulo operation to change interval
                if j % slice_point == 0:
                    test_xe_batches.append(test_x[i][j:j+self.seq_len_in])
                    test_xd_batches.append(test_y[i][j+self.seq_len_in - 1:j+self.seq_len_in+self.seq_len_out - 1])
                    test_y_batches.append(test_y[i][j + self.seq_len_in:j + self.seq_len_in + self.seq_len_out])

        test_xe_batches = np.stack(test_xe_batches, axis=0)
        test_xd_batches = np.stack(test_xd_batches, axis=0)
        test_y_batches = np.stack(test_y_batches, axis=0)

        return [test_xe_batches, test_xd_batches], test_y_batches

    def generate_training_batches(self):
        """
        Generate batch to be used in training
        :return: Batch for encoder and decoder inputs and a batch for output: [input encoder, input decoder], output_decoder
        """
        # Split into training set
        if hasattr(self.normalized_input_data, 'shape') and hasattr(self.normalized_output_data, 'shape'):
            train_x = self.normalized_input_data[:, :self.normalized_input_data.shape[1]//int((1 / self.test_train_ratio))]
            train_y = self.normalized_output_data[:, :self.normalized_output_data.shape[1] // int((1 / self.test_train_ratio))]
        else:
            train_x = []
            train_y = []
            for building in self.normalized_input_data:
                train_x.append(building[:int(len(building) * (1 - self.test_train_ratio))])
            for building in self.normalized_output_data:
                train_y.append(building[:int(len(building) * (1 - self.test_train_ratio))])

        while True:
            # Batch input for encoder
            batch_xe = []
            # Batch input for decoder for guided training
            batch_xd = []
            # Batch output
            batch_y = []

            for i in range(self.batch_size):
                # Select a random building from the training set
                bd = np.random.randint(0, self.normalized_input_data.shape[0])

                # Grab a random starting point from 0 to length of dataset - input length encoder - input length decoder
                sp = np.random.randint(0, len(train_x[bd]) - self.seq_len_in - self.seq_len_out)

                # Append samples to batches
                batch_xe.append(train_x[bd][sp:sp+self.seq_len_in])
                batch_xd.append(train_y[bd][sp+self.seq_len_in-1:sp+self.seq_len_in+self.seq_len_out-1])
                batch_y.append(train_y[bd][sp+self.seq_len_in:sp+self.seq_len_in+self.seq_len_out])

            # Stack batches and yield them
            batch_xe = np.stack(batch_xe)
            batch_xd = np.stack(batch_xd)
            batch_y = np.stack(batch_y)
            yield [batch_xe, batch_xd], batch_y

    def create_training_batch(self):
        """
        Generate single batch
        :return: Batch for encoder and decoder inputs and a batch for output
        """
        # Split into training set
        if hasattr(self.normalized_input_data, 'shape') and hasattr(self.normalized_output_data, 'shape'):
            train_x = self.normalized_input_data[:, :self.normalized_input_data.shape[1]//int((1 / self.test_train_ratio))]
            train_y = self.normalized_output_data[:, :self.normalized_output_data.shape[1] // int((1 / self.test_train_ratio))]
        else:
            train_x = []
            train_y = []
            for building in self.normalized_input_data:
                train_x.append(building[:int(len(building) * (1 - self.test_train_ratio))])
            for building in self.normalized_output_data:
                train_y.append(building[:int(len(building) * (1 - self.test_train_ratio))])

        # Batch input for encoder
        batch_xe = []
        # Batch input for decoder for guided training
        batch_xd = []
        # Batch output
        batch_y = []

        for i in range(self.batch_size):
            # Select a random building from the training set
            bd = np.random.randint(0, self.normalized_input_data.shape[0])

            # Grab a random starting point from 0 to length of dataset - input length encoder - input length decoder
            sp = np.random.randint(0, len(train_x[bd]) - self.seq_len_in - self.seq_len_out)

            # Append samples to batches
            batch_xe.append(train_x[bd][sp:sp+self.seq_len_in])
            batch_xd.append(train_y[bd][sp+self.seq_len_in-1:sp+self.seq_len_in+self.seq_len_out-1])
            batch_y.append(train_y[bd][sp+self.seq_len_in:sp+self.seq_len_in+self.seq_len_out])

        # Stack batches and yield them
        batch_xe = np.stack(batch_xe)
        batch_xd = np.stack(batch_xd)
        batch_y = np.stack(batch_y)

        return [batch_xe, batch_xd], batch_y

    def create_validation_sample(self):
        """
        Create a single validation sample, can be used to make predictions
        :return: Validation sample
        """
        # Split into testing set
        if hasattr(self.normalized_input_data, 'shape') and hasattr(self.normalized_output_data, 'shape'):
            test_x = self.normalized_input_data[:, -self.normalized_input_data.shape[1] // int((1 / self.test_train_ratio)):]
            test_y = self.normalized_output_data[:, -self.normalized_output_data.shape[1] // int((1 / self.test_train_ratio)):]
        else:
            test_x = []
            test_y = []
            for building in self.normalized_input_data:
                test_x.append(building[-int(len(building) * self.test_train_ratio):])
            for building in self.normalized_output_data:
                test_y.append(building[-int(len(building) * self.test_train_ratio):])

        # Batch input for encoder
        batch_xe = []
        # Batch input for decoder for guided training
        batch_xd = []
        # Batch output
        batch_y = []

        # Select a random building from the training set
        bd = np.random.randint(0, len(test_x))

        # Grab a random starting point from 0 to length of dataset - input length encoder - input length decoder
        sp = np.random.randint(0, len(test_x[bd]) - self.seq_len_in - self.seq_len_out)

        # Append sample to batch
        batch_xe.append(test_x[bd][sp:sp + self.seq_len_in])
        batch_xd.append(test_y[bd][sp + self.seq_len_in - 1:sp + self.seq_len_in + self.seq_len_out - 1])
        batch_y.append(test_y[bd][sp + self.seq_len_in:sp + self.seq_len_in + self.seq_len_out])

        # Output during input frames
        batch_y_prev = test_y[bd][sp:sp + self.seq_len_in]

        # Stack batches and return them
        batch_xe = np.stack(batch_xe)
        batch_xd = np.stack(batch_xd)
        batch_y = np.stack(batch_y)

        return [batch_xe, batch_xd], batch_y, batch_y_prev

    def create_validation_data_with_prev_y_steps(self, slice_point=1500):
        """
        Generate validation data of the whole testing set on a slicing point, including the previous y steps.
        :return: the validation data: [input encoder, input decoder], output_decoder
        """
        test_xe_batches = []
        test_xd_batches = []
        test_y_batches = []
        test_y_batches_prev = []

        # Split into testing set
        if hasattr(self.normalized_input_data, 'shape') and hasattr(self.normalized_output_data, 'shape'):
            test_x = self.normalized_input_data[:, -self.normalized_input_data.shape[1] // int((1 / self.test_train_ratio)):]
            test_y = self.normalized_output_data[:, -self.normalized_output_data.shape[1] // int((1 / self.test_train_ratio)):]
        else:
            test_x = []
            test_y = []
            for building in self.normalized_input_data:
                test_x.append(building[-int(np.shape(building)[0] * self.test_train_ratio):])
            for building in self.normalized_output_data:
                test_y.append(building[-int(np.shape(building)[0] * self.test_train_ratio):])

        for i in range(len(test_x)):
            for j in range(len(test_x[i]) - self.seq_len_out - self.seq_len_in):
                #  Change modulo operation to change interval
                if j % slice_point == 0:
                    test_xe_batches.append(test_x[i][j:j+self.seq_len_in])
                    test_xd_batches.append(test_y[i][j+self.seq_len_in - 1:j+self.seq_len_in+self.seq_len_out - 1])
                    test_y_batches.append(test_y[i][j + self.seq_len_in:j + self.seq_len_in + self.seq_len_out])
                    test_y_batches_prev.append(test_y[i][j:j + self.seq_len_in])

        test_xe_batches = np.stack(test_xe_batches, axis=0)
        test_xd_batches = np.stack(test_xd_batches, axis=0)
        test_y_batches = np.stack(test_y_batches, axis=0)
        test_y_batches_prev = np.stack(test_y_batches_prev, axis=0)

        return test_xe_batches, test_xd_batches, test_y_batches, test_y_batches_prev

    def create_training_sample(self):
        """
        Create a single training sample
        :return: Training sample
        """
        # Split into training set
        if hasattr(self.normalized_input_data, 'shape') and hasattr(self.normalized_output_data, 'shape'):
            train_x = self.normalized_input_data[:, :self.normalized_input_data.shape[1]//int((1 / self.test_train_ratio))]
            train_y = self.normalized_output_data[:, :self.normalized_output_data.shape[1] // int((1 / self.test_train_ratio))]
        else:
            train_x = []
            train_y = []
            for building in self.normalized_input_data:
                train_x.append(building[-int(len(building) * self.test_train_ratio):])
            for building in self.normalized_output_data:
                train_y.append(building[-int(len(building) * self.test_train_ratio):])

        # Batch input for encoder
        batch_xe = []
        # Batch input for decoder for guided training
        batch_xd = []
        # Batch output
        batch_y = []

        # Select a random building from the training set
        bd = np.random.randint(0, len(train_x))

        # Grab a random starting point from 0 to length of dataset - input length encoder - input length decoder
        sp = np.random.randint(0, len(train_x[bd]) - self.seq_len_in - self.seq_len_out)

        # Append sample to batch
        batch_xe.append(train_x[bd][sp:sp + self.seq_len_in])
        batch_xd.append(train_y[bd][sp + self.seq_len_in - 1:sp + self.seq_len_in + self.seq_len_out - 1])
        batch_y.append(train_y[bd][sp + self.seq_len_in:sp + self.seq_len_in + self.seq_len_out])

        # Output during input frames
        batch_y_prev = train_y[bd][sp:sp + self.seq_len_in]

        # Stack batches and return them
        batch_xe = np.stack(batch_xe)
        batch_xd = np.stack(batch_xd)
        batch_y = np.stack(batch_y)

        return [batch_xe, batch_xd], batch_y, batch_y_prev

    def train(self):
        """
        Train the model
        :return: Histories
        """
        histories = []
        val_losses = []
        train_losses = []

        # Path to save best model
        # model_saving_filepath = self.name + "-e{epoch:04d}-ss" + str(self.state_size) + "-vl{val_loss:.5f}.h5"
        model_saving_filepath = self.name + "-ss" + str(self.state_size) + "-agg" + self.agg_level + "-best_weights.h5"

        from keras.losses import mean_squared_error
        if "attention" in self.name:
            from tensorflow.python.keras.optimizers import Adam
            from tensorflow.python.keras.callbacks import ModelCheckpoint
            self.model.compile(Adam(self.learning_rate), mean_squared_error, metrics=self.validation_metrics)
            checkpoint = ModelCheckpoint(filepath=model_saving_filepath, monitor='val_loss', verbose=1,
                                         save_best_only=True,
                                         save_weights_only=True)
        else:
            from keras.callbacks import ModelCheckpoint
            from keras.optimizers import Adam
            self.model.compile(Adam(self.learning_rate), mean_squared_error, metrics=self.validation_metrics)
            checkpoint = ModelCheckpoint(filepath=model_saving_filepath, monitor='val_loss', verbose=1,
                                         save_best_only=True,
                                         save_weights_only=True)

        history = None

        # Set checkpoint for saving model
        callbacks_list = [checkpoint]

        for i in range(self.intermediates):
            try:
                history = self.model.fit_generator(self.generate_training_batches(),
                                                   steps_per_epoch=self.steps_per_epoch, epochs=self.epochs,
                                                   validation_data=self.validation_data, callbacks=callbacks_list)

                val_losses.extend(history.history['val_loss'])
                train_losses.extend(history.history['loss'])

                histories.append(history)
            except KeyboardInterrupt:
                print("Training interrupted!")

            # If given, plot the loss
            if self.plot_loss and history:
                plt.plot(history.history['loss'], label="loss")
                plt.plot(history.history['val_loss'], label="val_loss")
                plt.yscale('linear')
                plt.legend()
                plt.title(label=self.name + " loss")
                plt.show()

        # Write file with history of loss
        history_file = open("history-agg{0}-{1}-minvl{2:.4f}-minl{3:.4f}.pkl".format(self.agg_level,
                                                                                     self.name,
                                                                                     np.amin(val_losses),
                                                                                     np.amin(train_losses)), "wb")
        pickle.dump({"name": "{0}-agg{1}".format(self.name, self.agg_level), "train_losses": train_losses, "val_losses": val_losses}, history_file)

        # Return the history of the training session
        return histories
