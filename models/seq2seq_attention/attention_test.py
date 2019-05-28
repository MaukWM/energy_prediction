# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# import sys
# import os
# sys.path.append(os.getcwd())

import pickle
import numpy as np
import keras as ks
import matplotlib.pyplot as plt
from keras.utils import print_summary

import metrics
from tensorflow.python.keras.optimizers import Adam
from models.seq2seq_attention.seq2seq_attention import build_seq2seq_attention_model

from utils import load_data
from models.seq2seq.seq2seq import build_seq2seq_model
from models.seq2seq_1dconv.seq2seq_1dconv import build_seq2seq_1dconv_model

# Define some variables for generating batches
buildings = 15
batch_size = 2

# Define the amount of features in the input and the output
input_feature_amount = 1  # 83 without static indicators, 150 with.
output_feature_amount = 1

# Define size of states used by GRU
state_size = 1

# Input and output length sequence (24 * 4 = 96 15 minute intervals in 24 hours)
seq_len_in = 4
seq_len_out = 4

normalized_input_data, output_data = load_data()


def generate_batches_return():
    """
    Generate batch to be used in training
    :return: Batch for encoder and decoder inputs and a batch for output
    """

    batch_xe = []
    batch_xd = []
    batch_y = []

    for i in range(batch_size):
        sp = np.random.randint(1, 100 - seq_len_in - seq_len_out)
        data = np.arange(100)

        batch_xe.append(data[sp:sp+seq_len_in])
        batch_xd.append(data[sp + seq_len_in - 1:sp + seq_len_in + seq_len_out - 1])
        batch_y.append(data[sp + seq_len_in:sp + seq_len_in + seq_len_out])

    batch_xe = np.stack(batch_xe)
    batch_xd = np.stack(batch_xd)
    batch_y = np.stack(batch_y)

    batch_xe = np.reshape(batch_xe, (batch_size, seq_len_in, input_feature_amount))
    batch_xd = np.reshape(batch_xd, (batch_size, seq_len_out, output_feature_amount))
    batch_y = np.reshape(batch_y, (batch_size, seq_len_out, output_feature_amount))

    return [batch_xe, batch_xd], batch_y


def generate_validation_sample():
    """
    Generate batch to be used for validation, also return the previous ys so we can plot the input as well
    :return: Batch for encoder and decoder inputs and a batch for output
    """
    batch_xe = []
    batch_xd = []
    batch_y = []

    sp = np.random.randint(1, 100)
    data = np.arange(100)

    batch_xe.append(data[sp:sp + seq_len_in])
    batch_xd.append(data[sp + seq_len_in - 1:sp + seq_len_in + seq_len_out - 1])
    batch_y.append(data[sp + seq_len_in:sp + seq_len_in + seq_len_out])

    # Output during input frames
    batch_y_prev = data[sp:sp + seq_len_in]

    # Stack batches and return them
    batch_xe = np.stack(batch_xe)
    batch_xd = np.stack(batch_xd)
    batch_y = np.stack(batch_y)

    batch_xe = np.reshape(batch_xe, (1, seq_len_in, input_feature_amount))
    batch_xd = np.reshape(batch_xd, (1, seq_len_out, output_feature_amount))
    batch_y = np.reshape(batch_y, (1, seq_len_out, output_feature_amount))

    return [batch_xe, batch_xd], batch_y, batch_y_prev


def make_prediction(E, D, previous_timesteps_x, previous_y, n_output_timesteps):
    """
    Function to make a prediction from previous timesteps
    :param E: The encoder model
    :param D: The decoder model
    :param previous_timesteps_x: The previous timesteps x values
    :param previous_y: The last y of the input timesteps
    :param n_output_timesteps: The amount of output timesteps
    :return: An array with the predicted values
    """
    # Get the state from the Encoder using the previous timesteps for x
    # Expand the previous timesteps, we must make the input a batch (going from shape (100, 149) to (1, 100, 149))
    enc_outs, enc_last_state = E.predict(np.expand_dims(previous_timesteps_x, axis=0))
    dec_state = enc_last_state

    # Initialize the outputs on the previous y so we have something to feed the net
    # It might be neater to feed a start symbol instead
    dec_out = np.expand_dims(previous_y, axis=0)
    outputs = []
    for i in range(n_output_timesteps):
        dec_out, attention, dec_state = D.predict([enc_outs, dec_state, dec_out])
        outputs.append(dec_out)

    # Concatenate the outputs, as they are batches
    # For example, going from a list of (1,1,1) to one unit of (1,100,1)
    # So we take the 0th element from the batch which are our outputs
    return np.concatenate(outputs, axis=1)[0]


def train(encdecmodel, steps_per_epoch, epochs, learning_rate, intermediates=1, load_weights_path=None,
          save_weights=True, save_weights_path=None, plot_loss=True, plot_yscale='linear'):
    """
    Train the model
    :param encdecmodel: Encoder-decoder stack
    :param steps_per_epoch: How many steps per epoch should be done
    :param epochs: Amount of epochs
    :param validation_data: Validation data
    :param learning_rate: Learning rate
    :param intermediates: How many intermediate steps should be done
    :param load_weights_path: Path to weights that should be loaded
    :param save_weights: Whether to save the weights
    :param save_weights_path: Path to save the weights
    :param plot_loss: Whether to plot the loss
    :param plot_yscale: y_scale (log/linear)
    :return: Histories
    """
    histories = []

    # Load weights if path to weights is given
    if load_weights_path:
        encdecmodel.load_weights(load_weights_path)

    encdecmodel.compile(Adam(learning_rate), ks.losses.mean_squared_error, metrics=[metrics.mean_error,
                                                                                    # ks.losses.mean_absolute_error
                                                                                    ])

    for i in range(intermediates):
        try:
            x, y = generate_batches_return()
            encdecmodel.train_on_batch(x, y)

            l = encdecmodel.evaluate(x, y, batch_size=batch_size, verbose=0)

            # print(encdecmodel.get_weights())

            print(l)
            # histories.append(history)
        except KeyboardInterrupt:
            print("Training interrupted!")

    # Return the history of the training session
    return histories


def predict(encoder, decoder, enc_input, dec_input, actual_output, prev_output, plot=True):
    """

    :param encoder: Encoder model
    :param decoder: Decoder model
    :param enc_input: Input for the encoder
    :param dec_input: Input for the decoder
    :param actual_output: The actual output (during decoding time)
    :param prev_output: The previous output (during encoding time)
    :param plot: Boolean to indicate if the prediction should be plotted
    :return: Made predictions.
    """
    # Make a prediction on the given data
    predictions = make_prediction(encoder, decoder, enc_input[0, :seq_len_in], dec_input[0, 0:1],
                                  seq_len_out)

    if plot:
        # Concat the ys so we get a smooth line for the ys
        ys = np.concatenate([prev_output, np.reshape(actual_output[0], newshape=seq_len_out)])

        # Plot them
        plt.plot(range(0, seq_len_in + seq_len_out), ys, label="real")
        plt.plot(range(seq_len_in, seq_len_in + seq_len_out), predictions, label="predicted")

        # This code can be used if you want the ys to be mapped with 3 labels
        # plt.plot(range(0, seq_len_in), prev_output, label="previous_real")
        # plt.plot(range(seq_len_in, seq_len_in + seq_len_out), output[0], label="real")
        # plt.plot(range(seq_len_in, seq_len_in + seq_len_out), predictions, label="predicted")
        plt.legend()
        plt.show()

    return predictions


if __name__ == "__main__":
    # Build the model
    encoder, decoder, encdecmodel = build_seq2seq_attention_model(input_feature_amount=input_feature_amount,
                                                                  output_feature_amount=output_feature_amount,
                                                                  state_size=state_size, seq_len_in=seq_len_in,
                                                                  seq_len_out=seq_len_out)

    print_summary(encdecmodel, line_length=150)

    train(encdecmodel=encdecmodel, steps_per_epoch=20, epochs=5,
          learning_rate=0.0075, plot_yscale='linear', load_weights_path=None, intermediates=700)

    predict_x_batches, predict_y_batches, predict_y_batches_prev = generate_validation_sample()

    predict(encoder, decoder, predict_x_batches[0], predict_x_batches[1], predict_y_batches, predict_y_batches_prev)



