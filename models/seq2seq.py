import pickle

import metrics

from models.model import Model

import keras as ks
import numpy as np
import matplotlib.pyplot as plt

from utils import denormalize


class Seq2Seq(Model):

    def __init__(self, name, data_dict, batch_size, state_size, input_feature_amount, output_feature_amount,
                 seq_len_in, seq_len_out, plot_time_steps_view, steps_per_epoch, epochs, learning_rate, intermediates,
                 agg_level, load_weights_path=None, plot_loss=False):

        super().__init__(name, data_dict, batch_size, state_size, input_feature_amount, output_feature_amount,
                         seq_len_in, seq_len_out, plot_time_steps_view, steps_per_epoch, epochs, learning_rate,
                         intermediates, agg_level)

        # Build the model
        self.encoder, self.decoder, self.model = self.build_model()

        if load_weights_path:
            self.model.load_weights(load_weights_path)

        self.plot_loss = plot_loss

    def build_model(self):
        """
        Function to build the seq2seq model used.
        :return: Encoder model, decoder model (used for predicting) and full model (used for training).
        """
        # Define model inputs for the encoder/decoder stack
        x_enc = ks.Input(shape=(None, self.input_feature_amount), name="x_enc")
        x_dec = ks.Input(shape=(None, self.output_feature_amount), name="x_dec")

        # Define the encoder GRU, which only has to return a state
        _, state = ks.layers.GRU(self.state_size, return_state=True)(x_enc)

        # Define the decoder GRU and the Dense layer that will transform sequences of size 20 vectors to
        # a sequence of 1-long vectors of final predicted values
        dec_gru = ks.layers.GRU(self.state_size, return_state=True, return_sequences=True)
        # dec_dense2 = ks.layers.TimeDistributed(ks.layers.Dense(state_size, activation='relu'))
        dec_dense = ks.layers.TimeDistributed(ks.layers.Dense(self.output_feature_amount, activation='linear'))

        # Use these definitions to calculate the outputs of out encoder/decoder stack
        dec_intermediates, _ = dec_gru(x_dec, initial_state=state)
        # dec_intermediates = dec_dense2(dec_intermediates)
        dec_outs = dec_dense(dec_intermediates)

        # Define the encoder/decoder stack model
        encdecmodel = ks.Model(inputs=[x_enc, x_dec], outputs=dec_outs)

        # Define the encoder model
        E = ks.Model(inputs=x_enc, outputs=state)

        # Define a state_in model for the Decoder model (which will be used for prediction)
        state_in = ks.Input(shape=(self.state_size,), name="state")

        # Use the previously defined layers to calculate the new output value and state for the prediction model as well
        dec_intermediate, new_state = dec_gru(x_dec, initial_state=state_in)
        # dec_intermediate = dec_dense2(dec_intermediate)
        dec_out = dec_dense(dec_intermediate)

        # Define the decoder/prediction model
        D = ks.Model(inputs=[x_dec, state_in], outputs=[dec_out, new_state])
        return E, D, encdecmodel

    def make_prediction(self, previous_timesteps_x, previous_y):
        """
        Function to make a prediction from previous timesteps
        :param previous_timesteps_x: The previous timesteps x values
        :param previous_y: The last y of the input timesteps
        :return: An array with the predicted values
        """
        # Get the state from the Encoder using the previous timesteps for x
        # Expand the previous timesteps, we must make the input a batch (going from shape (100, 149) to (1, 100, 149))
        state = self.encoder.predict(np.expand_dims(previous_timesteps_x, axis=0))

        # Initialize the outputs on the previous y so we have something to feed the net
        # It might be neater to feed a start symbol instead
        outp = np.expand_dims(previous_y, axis=0)
        outputs = []
        for i in range(self.seq_len_out):
            outp, state = self.decoder.predict([outp, state])
            outputs.append(outp)

        # Concatenate the outputs, as they are batches
        # For example, going from a list of (1,1,1) to one unit of (1,100,1)
        # So we take the 0th element from the batch which are our outputs
        return np.concatenate(outputs, axis=1)[0]

    def predict(self, enc_input, dec_input, actual_output, prev_output, plot=True):
        """
        Make a prediction and plot the result
        :param enc_input: Input for the encoder
        :param dec_input: Input for the decoder
        :param actual_output: The actual output (during decoding time)
        :param prev_output: The previous output (during encoding time)
        :param plot: Boolean to indicate if the prediction should be plotted
        :return: Made normalized_predictions.
        """
        # Make a prediction on the given data
        normalized_predictions = self.make_prediction(enc_input[0, :self.seq_len_in], dec_input[0, 0:1])

        # Concat the normalized_ys so we get a smooth line for the normalized_ys
        normalized_ys = np.concatenate([prev_output, actual_output[0]])[-self.plot_time_steps_view:]

        ys = denormalize(normalized_ys, self.output_std, self.output_mean)
        predictions = denormalize(normalized_predictions, self.output_std, self.output_mean)

        if plot:
            # Plot them
            plt.plot(range(0, self.plot_time_steps_view), ys, label="real")
            plt.plot(range(self.plot_time_steps_view - self.seq_len_out, self.plot_time_steps_view), predictions,
                     label="predicted")
            plt.legend()
            plt.title(label="s2s")
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
    #     # Return the history of the training session
    #     return histories
