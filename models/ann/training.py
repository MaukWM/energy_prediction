import numpy as np
import keras as ks
import matplotlib.pyplot as plt
from keras.losses import mean_absolute_percentage_error

import metrics
from models.ann.ann import build_ann_model
from models.main import generate_batches, generate_validation_data, seq_len_in, seq_len_out, plot_last_time_steps_view, \
    state_size, input_feature_amount, output_feature_amount, validation_metrics, generate_validation_sample
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


def make_prediction(model, input):
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
    predictions = model.predict(input)

    # Concatenate the outputs, as they are batches
    # For example, going from a list of (1,1,1) to one unit of (1,100,1)
    # So we take the 0th element from the batch which are our outputs
    return np.reshape(predictions, newshape=(seq_len_out, output_feature_amount))


def train(model, steps_per_epoch, epochs, validation_data, learning_rate, intermediates=1, load_weights_path=None,
          save_weights=True, save_weights_path=None, plot_loss=True, plot_yscale='linear'):
    """
    Train the model
    :param model: Encoder-decoder stack
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
        model.load_weights(load_weights_path)

    model.compile(ks.optimizers.Adam(learning_rate), ks.losses.mean_squared_error, metrics=validation_metrics)

    for i in range(intermediates):
        try:
            history = model.fit_generator(generate_batches(), steps_per_epoch=steps_per_epoch, epochs=epochs,
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
                model.save_weights(save_weights_path)
            else:
                # Save weights with amount of epochs trained, loss and validation loss.
                model.save_weights("s2s-l{0}-tl{1:.3f}-vl{2:.3f}-i{3}-o{4}-e{5}-seq2seq.h5".format(str(learning_rate),
                                                                                                         history.history['loss'][-1],
                                                                                                         history.history['val_loss'][-1],
                                                                                                         seq_len_in,
                                                                                                         seq_len_out,
                                                                                                         epochs*intermediates))

    # Return the history of the training session
    return histories


def predict(model, input, actual_output, prev_output, plot=True):
    """

    :param model: The ANN model
    :param enc_input: Input for the encoder
    :param dec_input: Input for the decoder
    :param actual_output: The actual output (during decoding time)
    :param prev_output: The previous output (during encoding time)
    :param plot: Boolean to indicate if the prediction should be plotted
    :return: Made predictions.
    """
    # Make a prediction on the given data
    predictions = make_prediction(model, input)

    if plot:
        # Concat the ys so we get a smooth line for the ys
        ys = np.concatenate([prev_output, actual_output[0]])[-plot_last_time_steps_view:]

        # Plot them
        plt.plot(range(0, plot_last_time_steps_view), ys, label="real")
        plt.plot(range(plot_last_time_steps_view - seq_len_out, plot_last_time_steps_view), predictions, label="predicted")
        plt.legend()
        plt.show()

    return predictions


if __name__ == "__main__":
    test_x_batches, test_y_batches = generate_validation_data()

    # Build the model
    ann_model = build_ann_model(input_feature_amount=input_feature_amount,
                                output_feature_amount=output_feature_amount,
                                seq_len_in=seq_len_in, seq_len_out=seq_len_out)

    print(ann_model.summary())

    train(model=ann_model, steps_per_epoch=100, epochs=50, validation_data=(test_x_batches, test_y_batches),
          learning_rate=0.002, plot_yscale='linear', load_weights_path=None, intermediates=100)

    # ann_model.load_weights(filepath="/home/mauk/Workspace/energy_prediction/models/ann/s2s-l0.00075-tl1.974-vl1.821-i96-o96-e250-seq2seq.h5")

    predict_x_batches, predict_y_batches, predict_y_batches_prev = generate_validation_sample()

    calculate_accuracy(predict_x_batches, predict_y_batches, predict_y_batches_prev, ann_model)

    predict(ann_model, predict_x_batches, predict_y_batches, predict_y_batches_prev)
