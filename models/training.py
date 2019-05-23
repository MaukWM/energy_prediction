# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
import sys
import os
sys.path.append(os.getcwd())

import pickle
import numpy as np
import keras as ks
import matplotlib.pyplot as plt
from models import metrics

from utils import load_data
from models.seq2seq.seq2seq import build_seq2seq_model
from models.seq2seq_1dconv.seq2seq_1dconv import build_seq2seq_1dconv_model

# Define some variables for generating batches
buildings = 15
batch_size = 512

# Define the amount of features in the input and the output
input_feature_amount = 83  # 83 without static indicators, 150 with.
output_feature_amount = 1

# Define size of states used by GRU
state_size = 156

# Input and output length sequence (24 * 4 = 96 15 minute intervals in 24 hours)
seq_len_in = 96
seq_len_out = 96

normalized_input_data, output_data = load_data()


def generate_validation_data():
    """
    Generate validation data of the whole testing set.
    :return: the validation data
    """
    test_xe_batches = []
    test_xd_batches = []
    test_y_batches = []

    for i in range(len(normalized_input_data)):
        for j in range(len(normalized_input_data[i]) - seq_len_out - seq_len_in):
            #  Change modulo operation to change interval
            if j % 21 == 0:
                test_xe_batches.append(normalized_input_data[i][j:j+seq_len_in])
                test_xd_batches.append(output_data[i][j+seq_len_in - 1:j+seq_len_in+seq_len_out - 1])
                test_y_batches.append(output_data[i][j + seq_len_in:j + seq_len_in + seq_len_out])

    test_xe_batches = np.stack(test_xe_batches, axis=0)
    test_xd_batches = np.stack(test_xd_batches, axis=0)
    test_y_batches = np.stack(test_y_batches, axis=0)

    return [test_xe_batches, test_xd_batches], test_y_batches


def generate_batches():
    """
    Generate batch to be used in training
    :return: Batch for encoder and decoder inputs and a batch for output
    """
    while True:
        # Split into training and testing set
        train_x, train_y = normalized_input_data[:, :normalized_input_data.shape[1]//2], output_data[:, :output_data.shape[1]//2]

        # Batch input for encoder
        batch_xe = []
        # Batch input for decoder for guided training
        batch_xd = []
        # Batch output
        batch_y = []

        for i in range(batch_size):
            # Select a random building from the training set
            bd = np.random.randint(0, buildings)

            # Grab a random starting point from 0 to length of dataset - input length encoder - input length decoder
            sp = np.random.randint(0, len(train_x[bd]) - seq_len_in - seq_len_out)

            # Append samples to batches
            batch_xe.append(train_x[bd][sp:sp+seq_len_in])
            batch_xd.append(train_y[bd][sp+seq_len_in-1:sp+seq_len_in+seq_len_out-1])
            batch_y.append(train_y[bd][sp+seq_len_in:sp+seq_len_in+seq_len_out])

        # Stack batches and yield them
        batch_xe = np.stack(batch_xe)
        batch_xd = np.stack(batch_xd)
        batch_y = np.stack(batch_y)
        yield [batch_xe, batch_xd], batch_y


def generate_validation_sample():
    """
    Generate batch to be used for validation, also return the previous ys so we can plot the input as well
    :return: Batch for encoder and decoder inputs and a batch for output
    """
    # Split into training and testing set
    test_x, test_y = normalized_input_data[:, normalized_input_data.shape[1]//2:], output_data[:, output_data.shape[1]//2:]

    # Batch input for encoder
    batch_xe = []
    # Batch input for decoder for guided training
    batch_xd = []
    # Batch output
    batch_y = []

    # Select a random building from the training set
    bd = np.random.randint(0, len(test_x))

    # Grab a random starting point from 0 to length of dataset - input length encoder - input length decoder
    sp = np.random.randint(0, len(test_x[bd]) - seq_len_in - seq_len_out)

    # Append sample to batch
    batch_xe.append(test_x[bd][sp:sp + seq_len_in])
    batch_xd.append(test_y[bd][sp + seq_len_in - 1:sp + seq_len_in + seq_len_out - 1])
    batch_y.append(test_y[bd][sp + seq_len_in:sp + seq_len_in + seq_len_out])

    # Output during input frames
    batch_y_prev = test_y[bd][sp:sp + seq_len_in]

    # Stack batches and return them
    batch_xe = np.stack(batch_xe)
    batch_xd = np.stack(batch_xd)
    batch_y = np.stack(batch_y)
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
    state = E.predict(np.expand_dims(previous_timesteps_x, axis=0))

    # Initialize the outputs on the previous y so we have something to feed the net
    # It might be neater to feed a start symbol instead
    outp = np.expand_dims(previous_y, axis=0)
    outputs = []
    for i in range(n_output_timesteps):
        outp, state = D.predict([outp, state])
        outputs.append(outp)

    # Concatenate the outputs, as they are batches
    # For example, going from a list of (1,1,1) to one unit of (1,100,1)
    # So we take the 0th element from the batch which are our outputs
    return np.concatenate(outputs, axis=1)[0]


def train(encdecmodel, steps_per_epoch, epochs, validation_data, learning_rate, intermediates=1, load_weights_path=None,
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

    for i in range(intermediates):
        try:
            encdecmodel.compile(ks.optimizers.Adam(learning_rate), ks.losses.mean_squared_error, metrics=[metrics.mean_error,
                                                                                                          # ks.losses.mean_absolute_error
                                                                                                          ])
            history = encdecmodel.fit_generator(generate_batches(), steps_per_epoch=steps_per_epoch, epochs=epochs,
                                                validation_data=validation_data)
            histories.append(history)
        except KeyboardInterrupt:
            print("Training interrupted!")

        # If given, plot the loss
        if plot_loss and history:
            plt.plot(history.history['loss'], label="loss")
            plt.plot(history.history['val_loss'], label="val_loss")
            plt.yscale(plot_yscale)
            plt.legend()
            plt.show()

        # Save weights
        if save_weights:
            if save_weights_path:
                encdecmodel.save_weights(save_weights_path)
            else:
                # Save weights with amount of epochs trained, loss and validation loss.
                encdecmodel.save_weights("l{0}-ss{1}-tl{2:.3f}-vl{3:.3f}-i{4}-o{5}-e{6}-seq2seq.h5".format(str(learning_rate),
                                                                                        str(state_size),
                                                                                        history.history['loss'][-1],
                                                                                        history.history['val_loss'][-1],
                                                                                        seq_len_in,
                                                                                        seq_len_out,
                                                                                        epochs*intermediates))

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
        ys = np.concatenate([prev_output, actual_output[0]])

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
    test_x_batches, test_y_batches = generate_validation_data()

    print(test_x_batches)
    print("=============================\n"
          "=============================\n"
          "=============================")
    print(test_y_batches)

    # # Build the model
    # encoder, decoder, encdecmodel = build_seq2seq_model(input_feature_amount=input_feature_amount,
    #                                                     output_feature_amount=output_feature_amount,
    #                                                     state_size=state_size, use_noise=False)

    # Build the model
    encoder, decoder, encdecmodel = build_seq2seq_1dconv_model(input_feature_amount=input_feature_amount,
                                                               output_feature_amount=output_feature_amount,
                                                               state_size=state_size, seq_len_in=seq_len_in,
                                                               use_noise=False)

    encdecmodel.summary()

    # print(normalized_input_data.shape)
    # print(test_y_batches.shape)
    # print(np.array(test_x_batches).shape)

    for a in generate_batches():
        (xe, xd), y = a
        print(xe.shape, xd.shape, y.shape)
        break

    train(encdecmodel=encdecmodel, steps_per_epoch=35, epochs=10, validation_data=(test_x_batches, test_y_batches),
          learning_rate=0.00075, plot_yscale='linear', load_weights_path=None, intermediates=3)

    # encdecmodel.load_weights(filepath="l0.00065-ss156-tl0.285-vl0.997-i192-o96-e420-seq2seq.h5")

    predict_x_batches, predict_y_batches, predict_y_batches_prev = generate_validation_sample()

    predict(encoder, decoder, predict_x_batches[0], predict_x_batches[1], predict_y_batches, predict_y_batches_prev)



