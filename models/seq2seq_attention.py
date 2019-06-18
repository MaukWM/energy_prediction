from tensorflow.python.keras.layers import Input, GRU, Dense, Concatenate, TimeDistributed
from tensorflow.python.keras.models import Model as tsModel
from tensorflow.python.keras.optimizers import Adam

import metrics
from layers.attention import AttentionLayer

from models.model import Model

import numpy as np
import matplotlib.pyplot as plt

from utils import denormalize, plot_attention_weights


class Seq2SeqAttention(Model):

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
        x_enc = Input(shape=(self.seq_len_in, self.input_feature_amount), name="x_enc")
        x_dec = Input(shape=(self.seq_len_out, self.output_feature_amount), name="x_dec")

        # Define the encoder GRU, which only has to return a state
        encoder_gru = GRU(self.state_size, return_sequences=True, return_state=True, name="encoder_gru")
        encoder_out, encoder_state = encoder_gru(x_enc)

        # Decoder GRU
        decoder_gru = GRU(self.state_size, return_state=True, return_sequences=True,
                          name="decoder_gru")
        # Use these definitions to calculate the outputs of out encoder/decoder stack
        dec_intermediates, decoder_state = decoder_gru(x_dec, initial_state=encoder_state)

        # Define the attention layer
        attn_layer = AttentionLayer(name="attention_layer")
        attn_out, attn_states = attn_layer([encoder_out, dec_intermediates])

        # Concatenate decoder and attn out
        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([dec_intermediates, attn_out])

        # Define the dense layer
        dense = Dense(self.output_feature_amount, activation='linear', name='output_layer')
        dense_time = TimeDistributed(dense, name='time_distributed_layer')
        decoder_pred = dense_time(decoder_concat_input)

        # Define the encoder/decoder stack model
        encdecmodel = tsModel(inputs=[x_enc, x_dec], outputs=decoder_pred)

        # Define the separate encoder model for inferencing
        encoder_inf_inputs = Input(shape=(self.seq_len_in, self.input_feature_amount), name="encoder_inf_inputs")
        encoder_inf_out, encoder_inf_state = encoder_gru(encoder_inf_inputs)
        encoder_model = tsModel(inputs=encoder_inf_inputs, outputs=[encoder_inf_out, encoder_inf_state])

        # Define the separate encoder model for inferencing
        decoder_inf_inputs = Input(shape=(1, self.output_feature_amount), name="decoder_inputs")
        encoder_inf_states = Input(shape=(self.seq_len_in, self.state_size), name="encoder_inf_states")
        decoder_init_state = Input(shape=(self.state_size,), name="decoder_init")

        decoder_inf_out, decoder_inf_state = decoder_gru(decoder_inf_inputs, initial_state=decoder_init_state)
        attn_inf_out, attn_inf_states = attn_layer([encoder_inf_states, decoder_inf_out])
        decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_inf_out, attn_inf_out])
        decoder_inf_pred = TimeDistributed(dense)(decoder_inf_concat)
        decoder_model = tsModel(inputs=[encoder_inf_states, decoder_init_state, decoder_inf_inputs],
                                outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_state])

        return encoder_model, decoder_model, encdecmodel

    def make_prediction(self, previous_timesteps_x, previous_y):
        """
        Function to make a prediction from previous timesteps
        :param previous_timesteps_x: The previous timesteps x values
        :param previous_y: The last y of the input timesteps
        :param n_output_timesteps: The amount of output timesteps
        :return: An array with the predicted values
        """
        enc_outs, enc_last_state = self.encoder.predict(np.expand_dims(previous_timesteps_x, axis=0))
        dec_state = enc_last_state

        # Initialize the outputs on the previous y so we have something to feed the net
        # It might be neater to feed a start symbol instead
        dec_out = np.expand_dims(previous_y, axis=0)
        outputs = []
        attention_weights = []
        for i in range(self.seq_len_out):
            dec_out, attention, dec_state = self.decoder.predict([enc_outs, dec_state, dec_out])
            outputs.append(dec_out)

            # Add attention weights
            attention_weights.append(attention)

        # Reshape and transpose attention weights so they make more sense
        attention_weights = np.reshape(np.stack(attention_weights), newshape=(self.seq_len_out, self.seq_len_in)).transpose()

        # Concatenate the outputs, as they are batches
        # For example, going from a list of (1,1,1) to one unit of (1,100,1)
        # So we take the 0th element from the batch which are our outputs
        return np.concatenate(outputs, axis=1)[0], attention_weights

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
        normalized_predictions, attention_weights = self.make_prediction(enc_input[0, :self.seq_len_in], dec_input[0, 0:1])

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
            plt.title(label=self.name)
            plt.show()

            plot_attention_weights(attention_weights)

        return normalized_predictions

    def calculate_accuracy(self, predict_x_batches, predict_y_batches):
        self.model.compile(Adam(1), metrics.root_mean_squared_error)

        eval_loss = self.model.evaluate(predict_x_batches, predict_y_batches, batch_size=1, verbose=1)

        real = predict_y_batches[0]

        real_mean = np.mean(real)

        # Calcluted with the min and max
        nrmsem = eval_loss / (np.amax(real) - np.amin(real))
        # Calculated with the mean
        nrmsea = eval_loss / real_mean

        print(self.name, "normalized root-mean-square deviation (max-min): {0:.2f}%".format(nrmsem * 100))
        print(self.name, "normalized root-mean-square deviation (mean): {0:.2f}%".format(nrmsea * 100))


