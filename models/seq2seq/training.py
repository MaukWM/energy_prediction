import numpy as np
import keras as ks
import matplotlib.pyplot as plt
from keras.losses import mean_absolute_percentage_error

import metrics
from models.main import generate_batches, generate_validation_data, seq_len_in, seq_len_out, plot_last_time_steps_view, \
    state_size, input_feature_amount, output_feature_amount
from models.seq2seq.seq2seq import build_seq2seq_model

# # Define some variables for generating batches
# buildings = 15
# batch_size = 256
#
# # Define the amount of features in the input and the output
# input_feature_amount = 83  # 83 without static indicators, 150 with.
# output_feature_amount = 1
#
# # Define size of states used by GRU
# state_size = 96
#
# # Input and output length sequence (24 * 4 = 96 15 minute intervals in 24 hours)
# seq_len_in = 96 * 2
# seq_len_out = 96
#
# plot_last_time_steps_view = 96 * 2


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
                                                                                            # mean_absolute_percentage_error
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
        ys = np.concatenate([prev_output, actual_output[0]])[-plot_last_time_steps_view:]

        # Plot them
        plt.plot(range(0, plot_last_time_steps_view), ys, label="real")
        plt.plot(range(plot_last_time_steps_view - seq_len_out, plot_last_time_steps_view), predictions, label="predicted")
        plt.legend()
        plt.show()

    return predictions


def calculate_accuracy(predict_x_batches, predict_y_batches, predict_y_batches_prev, encdecmodel):
    encdecmodel.compile(ks.optimizers.Adam(1), metrics.root_mean_squared_error)

    eval_loss = encdecmodel.evaluate(predict_x_batches, predict_y_batches,
                                     batch_size=1, verbose=1)

    predictions = predict(encoder, decoder, predict_x_batches[0], predict_x_batches[1], predict_y_batches, predict_y_batches_prev, plot=False)

    real = predict_y_batches[0]

    real_mean = np.mean(real)

    rrse_upper = np.sum(np.square(np.subtract(predictions, real)))
    rrse_lower = np.sum(np.square(np.subtract(np.full(predictions.size, real_mean), real)))
    rrse = rrse_upper / rrse_lower

    nrmsem = eval_loss / (np.amax(real) - np.amin(real))
    nrmsea = eval_loss / real_mean

    print("Loss: {}".format(eval_loss))
    print("Real mean: {}".format(real_mean))
    print("RRSE: {}%".format(rrse * 100))
    print("NRMSEM: {}%".format(nrmsem * 100))
    print("NRMSEA: {}%".format(nrmsea * 100))


if __name__ == "__main__":
    test_x_batches, test_y_batches = generate_validation_data()

    # Build the model
    encoder, decoder, encdecmodel = build_seq2seq_model(input_feature_amount=input_feature_amount,
                                                        output_feature_amount=output_feature_amount,
                                                        state_size=state_size)

    train(encdecmodel=encdecmodel, steps_per_epoch=150, epochs=100, validation_data=(test_x_batches, test_y_batches),
          learning_rate=0.00045, plot_yscale='linear', load_weights_path=None, intermediates=100)

    # encdecmodel.load_weights(filepath="/home/mauk/Workspace/energy_prediction/models/seq2seq_1dconv/256ss-4conv-layers/l0.00025-ss256-tl0.045-vl0.660-i480-o96-e6000-seq2seq.h5")

    # predict_x_batches, predict_y_batches, predict_y_batches_prev = generate_validation_sample()

    # calculate_accuracy(predict_x_batches, predict_y_batches, predict_y_batches_prev, encdecmodel)

    # predict(encoder, decoder, predict_x_batches[0], predict_x_batches[1], predict_y_batches, predict_y_batches_prev)
