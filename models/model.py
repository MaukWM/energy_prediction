from metrics import mean_error

import numpy as np


class Model:

    def __init__(self, data_dict, batch_size=256, state_size=42, input_feature_amount=83, output_feature_amount=1,
                 seq_len_in=96, seq_len_out=96, plot_time_steps_view=192):
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

        self.validation_metrics = [mean_error]
        self.test_train_ratio = 0.5

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
