# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
import pickle
import numpy as np
import keras as ks
import matplotlib.pyplot as plt
import metrics

# Define some variables for generating batches
buildings = 7
batch_size = 512

# Define the amount of features in the input and the output
input_feature_amount = 149
output_feature_amount = 1

# Define size of states used by GRU
state_size = 32

# Input and output length sequence (24 * 4 = 96 15 minute intervals in 24 hours)
seq_len_in = 96
seq_len_out = 96

# # Load in the prepared data
# input_data = open("../data/prepared/input_data.pkl", "rb")
# normalized_input_data, output_data = pickle.load(input_data)
#
# # Define the input and output of the testing set
# test_x, test_y = normalized_input_data[:, normalized_input_data.shape[1]//2:], output_data[:, output_data.shape[1]//2:]
#
# # Change the length so we can generate batches from test_x and test_y
# new_len = (seq_len_in + seq_len_out) * (test_x.shape[1] // (seq_len_in + seq_len_out))
# test_x, test_y = test_x[:, :new_len], test_y[:, :new_len]
#
# # Make them batches
# test_x_batches = np.reshape(test_x, (-1, (seq_len_in + seq_len_out), input_feature_amount))
# test_y_batches = np.reshape(test_y, (-1, (seq_len_in + seq_len_out), output_feature_amount))
#
# # Set the batches
# test_xe_batches = test_x_batches[:, :seq_len_in]
# test_xd_batches = test_y_batches[:, seq_len_in-1:-1]
# test_x_batches = [test_xe_batches, test_xd_batches]
# test_y_batches = test_y_batches[:, seq_len_in:]


def generate_validation_data():
    """
    Generate validation data of the whole testing set.
    :return: the validation data
    """
    input_data = open("../data/prepared/input_data.pkl", "rb")
    normalized_input_data, output_data = pickle.load(input_data)

    print(np.shape(normalized_input_data))

    test_xe_batches = []
    test_xd_batches = []
    test_y_batches = []
    # test_y_batches_prev = []

    for i in range(len(normalized_input_data)):
        for j in range(len(normalized_input_data[i]) - seq_len_out - seq_len_in):
            #  Change modulo operation to change interval
            if j % 16 == 0:
                test_xe_batches.append(normalized_input_data[i][j:j+seq_len_in])
                test_xd_batches.append(output_data[i][j+seq_len_in - 1:j+seq_len_in+seq_len_out - 1])
                test_y_batches.append(output_data[i][j + seq_len_in:j + seq_len_in + seq_len_out])
                # test_y_batches_prev.append(output_data[i][j: j + seq_len_in])

    # normal_shape = None
    # for e in test_xe_batches:
    #     if normal_shape is None:
    #         normal_shape = e.shape
    #     if e.shape != normal_shape:
    #         raise Exception("HET IS STUK", normal_shape, e.shape)

    test_xe_batches = np.stack(test_xe_batches, axis=0)
    test_xd_batches = np.stack(test_xd_batches, axis=0)
    test_y_batches = np.stack(test_y_batches, axis=0)
    # test_y_batches_prev = np.stack(test_y_batches, axis=0)

    return [test_xe_batches, test_xd_batches], test_y_batches # , test_y_batches_prev


def generate_batches():
    """
    Generate batch to be used in training
    :return: Batch for encoder and decoder inputs and a batch for output
    """
    # Read data
    input_data = open("../data/prepared/input_data.pkl", "rb")
    normalized_input_data, output_data = pickle.load(input_data)

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
    # Read data
    input_data = open("../data/prepared/input_data.pkl", "rb")
    normalized_input_data, output_data = pickle.load(input_data)

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


def build_seq2seq_model(use_noise=False):
    """
    Function to build the seq2seq model used.
    :return: Encoder model, decoder model (used for predicting) and full model (used for training).
    """
    # Define model inputs for the encoder/decoder stack
    x_enc = ks.Input(shape=(None, input_feature_amount), name="x_enc")
    x_dec = ks.Input(shape=(None, output_feature_amount), name="x_dec")

    if use_noise:
        x_dec_t = ks.layers.GaussianNoise(0.2)(x_dec)
    else:
        x_dec_t = x_dec

    # Define the encoder GRU, which only has to return a state
    _, state = ks.layers.GRU(state_size, return_state=True)(x_enc)

    # Define the decoder GRU and the Dense layer that will transform sequences of size 20 vectors to
    # a sequence of 1-long vectors of final predicted values
    dec_gru = ks.layers.GRU(state_size, return_state=True, return_sequences=True)
    # dec_dense2 = ks.layers.TimeDistributed(ks.layers.Dense(state_size, activation='relu'))
    dec_dense = ks.layers.TimeDistributed(ks.layers.Dense(output_feature_amount, activation='linear'))

    # Use these definitions to calculate the outputs of out encoder/decoder stack
    dec_intermediates, _ = dec_gru(x_dec_t, initial_state=state)
    # dec_intermediates = dec_dense2(dec_intermediates)
    dec_outs = dec_dense(dec_intermediates)

    # Define the encoder/decoder stack model
    encdecmodel = ks.Model(inputs=[x_enc, x_dec], outputs=dec_outs)

    # Define the encoder model
    E = ks.Model(inputs=x_enc, outputs=state)

    # Define a state_in model for the Decoder model (which will be used for prediction)
    state_in = ks.Input(shape=(state_size,), name="state")

    # Use the previously defined layers to calculate the new output value and state for the prediction model as well
    dec_intermediate, new_state = dec_gru(x_dec, initial_state=state_in)
    # dec_intermediate = dec_dense2(dec_intermediate)
    dec_out = dec_dense(dec_intermediate)

    # Define the decoder/prediction model
    D = ks.Model(inputs=[x_dec, state_in], outputs=[dec_out, new_state])
    return E, D, encdecmodel


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

    # Build the model
    encoder, decoder, encdecmodel = build_seq2seq_model(use_noise=False)

    # train(encdecmodel=encdecmodel, steps_per_epoch=25, epochs=10, validation_data=(test_x_batches, test_y_batches),
    #       learning_rate=0.00075, plot_yscale='linear', load_weights_path=None, intermediates=3)

    encdecmodel.load_weights(filepath="l0.00075-ss32-tl0.490-vl0.483-i96-o96-e30-seq2seq.h5")

    predict_x_batches, predict_y_batches, predict_y_batches_prev = generate_validation_sample()

    predict(encoder, decoder, predict_x_batches[0], predict_x_batches[1], predict_y_batches, predict_y_batches_prev)



