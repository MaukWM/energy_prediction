from keras.layers import Flatten, Reshape

import metrics

from models.model import Model

import keras as ks
import numpy as np
import matplotlib.pyplot as plt

from utils import denormalize


class Ann(Model):

    def __init__(self, name, data_dict, batch_size, state_size, input_feature_amount, output_feature_amount,
                 seq_len_in, seq_len_out, plot_time_steps_view, steps_per_epoch, epochs, learning_rate, intermediates,
                 agg_level, load_weights_path=None, plot_loss=False):

        super().__init__(name, data_dict, batch_size, state_size, input_feature_amount, output_feature_amount,
                         seq_len_in, seq_len_out, plot_time_steps_view, steps_per_epoch, epochs, learning_rate,
                         intermediates, agg_level)

        # Build the model
        self.model = self.build_model()

        if load_weights_path:
            self.model.load_weights(load_weights_path)

        self.plot_loss = plot_loss

    def build_model(self):
        """
        Function to build the ann model.
        :return: Encoder model, decoder model (used for predicting) and full model (used for training).
        """

        input_sequence = ks.Input(shape=(self.seq_len_in, self.input_feature_amount), name="input")
        not_used = ks.Input(shape=(None, self.output_feature_amount), name="not_used")

        input_flat = Flatten()(input_sequence)

        dense1 = ks.layers.Dense(1024, activation='relu')
        dense1_out = dense1(input_flat)

        dense2 = ks.layers.Dense(256, activation='relu')
        dense2_out = dense2(dense1_out)

        dense3 = ks.layers.Dense(self.seq_len_out * self.output_feature_amount, activation='linear')
        dense3_out = dense3(dense2_out)
        seq_out = Reshape((self.seq_len_out, self.output_feature_amount))(dense3_out)

        model = ks.Model(inputs=[input_sequence, not_used], outputs=seq_out)

        return model

    def make_prediction(self, input):
        """
        Function to make a prediction from previous timesteps
        :param: the input
        :return: An array with the predicted values
        """
        # Make prediction
        predictions = self.model.predict(input)

        # Concatenate the outputs, as they are batches
        # For example, going from a list of (1,1,1) to one unit of (1,100,1)
        # So we take the 0th element from the batch which are our outputs
        return np.reshape(predictions, newshape=(self.seq_len_out, self.output_feature_amount))

    # (self, enc_input, dec_input, actual_output, prev_output):
    def predict(self, enc_input, dec_input, actual_output, prev_output, plot=True):
        """
        Make a prediction and plot the result
        :param enc_input: Input for the encoder
        :param dec_input: The actual output
        :param actual_output: The previous output
        :return: Made normalized_predictions.
        """
        # Make a prediction on the given data
        normalized_predictions = self.make_prediction([enc_input, dec_input])

        # Concat the normalized_ys so we get a smooth line for the normalized_ys
        normalized_ys = np.concatenate([actual_output[0], dec_input[0]])[-self.plot_time_steps_view:]

        ys = denormalize(normalized_ys, self.output_std, self.output_mean)
        predictions = denormalize(normalized_predictions, self.output_std, self.output_mean)

        if plot:
            # Plot them
            plt.plot(range(0, self.plot_time_steps_view), ys, label="real")
            plt.plot(range(self.plot_time_steps_view - self.seq_len_out, self.plot_time_steps_view), predictions,
                     label="predicted")
            plt.legend()
            plt.title(label="ann")
            plt.show()

        return normalized_predictions

    def calculate_accuracy(self, predict_x_batches, predict_y_batches):
        self.model.compile(ks.optimizers.Adam(1), metrics.root_mean_squared_error)

        eval_loss = self.model.evaluate(predict_x_batches, predict_y_batches, batch_size=1, verbose=1)

        real = predict_y_batches[0]

        real_mean = np.mean(real)

        # Calcluted with the min and max
        nrmsem = eval_loss / (np.amax(real) - np.amin(real))
        # Calculated with the mean
        nrmsea = eval_loss / real_mean

        print(self.name, "normalized root-mean-square deviation (max-min): {0:.2f}%".format(nrmsem * 100))
        print(self.name, "normalized root-mean-square deviation (mean): {0:.2f}%".format(nrmsea * 100))

    # def train(self):
    #     """
    #     Train the model
    #     :return: Histories
    #     """
    #     histories = []
    #     val_losses = []
    #     losses_dict = []
    #
    #     self.model.compile(ks.optimizers.Adam(self.learning_rate), ks.losses_dict.mean_squared_error,
    #                        metrics=self.validation_metrics)
    #
    #     history = None
    #
    #     for i in range(self.intermediates):
    #         try:
    #             history = self.model.fit_generator(self.generate_training_batches(),
    #                                                steps_per_epoch=self.steps_per_epoch, epochs=self.epochs,
    #                                                validation_data=self.validation_data)
    #
    #             self.model.save_weights(
    #                 self.name + "-l{0}-ss{1}-tl{2:.4f}-vl{3:.4f}-i{4}-o{5}.h5".format(str(self.learning_rate),
    #                                                                          str(self.state_size),
    #                                                                          history.history['loss'][-1],
    #                                                                          history.history['val_loss'][-1],
    #                                                                          self.seq_len_in,
    #                                                                          self.seq_len_out))
    #
    #             val_losses.extend(history.history['val_loss'])
    #             losses_dict.extend(history.history['loss'])
    #
    #             histories.append(history)
    #         except KeyboardInterrupt:
    #             self.model.save_weights(
    #                 self.name + "-l{0}-ss{1}-interrupted-i{2}-o{3}.h5".format(str(self.learning_rate),
    #                                                                  str(self.state_size),
    #                                                                  self.seq_len_in,
    #                                                                  self.seq_len_out))
    #             print("Training interrupted!")
    #
    #         # If given, plot the loss
    #         if self.plot_loss and history:
    #             plt.plot(history.history['loss'], label="loss")
    #             plt.plot(history.history['val_loss'], label="val_loss")
    #             plt.yscale('linear')
    #             plt.legend()
    #             plt.title(label=self.name + " loss")
    #             plt.show()
    #
    #     # Write file with history of loss
    #     history_file = open("{0}-minvl{1:.4f}-minl{2:.4f}-history.pkl".format(self.name,
    #                                                                           np.amin(val_losses),
    #                                                                           np.amin(losses_dict)), "wb")
    #     pickle.dump({"losses_dict": losses_dict, "val_losses": val_losses}, history_file)
    #
    #     return histories
