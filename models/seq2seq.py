# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
import pickle
import numpy as np
import keras as ks
import matplotlib.pyplot as plt

# Define some variables for generating batches
buildings = 7
batch_size = 128

# Define the amount of features in the input and the output
input_feature_amount = 149
output_feature_amount = 1

# Define size of states used by GRU
state_size = 32

# Input and output length sequence (24 * 4 = 96 15 minute intervals in 24 hours)
seq_len_in = 96
seq_len_out = 96


def generate_batches():
    """
    Generate batch to be used in training
    :return: Batch for encoder and decoder inputs and a batch for output
    """
    # Read data
    input_data = open("../data/prepared/input_data.pkl", "rb")
    normalized_input_data, output_data = pickle.load(input_data)

    # Split into training and testing set
    train_x, train_y = normalized_input_data[:, :normalized_input_data.shape[1]//2], output_data[:, :output_data.shape[1]//2]
    test_x, test_y = normalized_input_data[:, normalized_input_data.shape[1]//2:], output_data[:, output_data.shape[1]//2:]

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
        batch_xd.append(train_x[bd][sp+seq_len_in:sp+seq_len_in+seq_len_out])
        batch_y.append(train_y[bd][sp+seq_len_in:sp+seq_len_in+seq_len_out])

    # Stack batches and yield them
    batch_xe = np.stack(batch_xe)
    batch_xd = np.stack(batch_xd)
    batch_y = np.stack(batch_y)
    # print(np.shape(batch_xe))
    # print(np.shape(batch_xd))
    # print(np.shape(batch_y))
    yield [batch_xe, batch_xd], batch_y


def build_seq2seq_model():
    """
    Function to build the seq2seq model used.
    :return: Encoder model, decoder model (used for predicting) and full model (used for training).
    """
    # Define model inputs for the encoder/decoder stack
    x_enc = ks.Input(shape=(None, input_feature_amount), name="x_enc")
    x_dec = ks.Input(shape=(None, output_feature_amount), name="x_dec")

    # Define the encoder GRU, which only has to return a state
    _, state = ks.layers.GRU(state_size, return_state=True)(x_enc)

    # Define the decoder GRU and the Dense layer that will transform sequences of size 20 vectors to
    # a sequence of 1-long vectors of final predicted values
    dec_gru = ks.layers.GRU(state_size, return_state=True, return_sequences=True)
    dec_dense = ks.layers.TimeDistributed(ks.layers.Dense(output_feature_amount, activation='linear'))

    # Use these definitions to calculate the outputs of out encoder/decoder stack
    dec_intermediates, _ = dec_gru(x_dec, initial_state=state)
    dec_outs = dec_dense(dec_intermediates)

    # Define the encoder/decoder stack model
    encdecmodel = ks.Model(inputs=[x_enc, x_dec], outputs=dec_outs)

    # Define the encoder model
    E = ks.Model(inputs=x_enc, outputs=state)

    # Define a state_in model for the Decoder model (which will be used for prediction)
    state_in = ks.Input(shape=(state_size,), name="state")

    # Use the previously defined layers to calculate the new output value and state for the prediction model as well
    dec_intermediate, new_state = dec_gru(x_dec, initial_state=state_in)
    dec_out = dec_dense(dec_intermediate)

    # Define the decoder/prediction model
    D = ks.Model(inputs=[x_dec, state_in], outputs=[dec_out, new_state])
    return E, D, encdecmodel


def make_prediction(E, D, previous_timesteps_x, previous_y, n_output_timesteps):
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


if __name__ == "__main__":
    encoder, decoder, encdecmodel = build_seq2seq_model()

    # encdecmodel.load_weights("model_weights.h5")

    encdecmodel.compile(ks.optimizers.Adam(0.03), ks.losses.mean_squared_error)
    encdecmodel.fit_generator(generate_batches(), steps_per_epoch=100, epochs=2)

    encdecmodel.save_weights("seq2seq_model_weights.h5")

    x, y = generate_batches().__next__()

    print(x[0], y[0])

    # plt.plot(x[:, 0])
    # plt.plot(x[:, 1])
    plt.plot(y)

    predictions = make_prediction(encoder, decoder, x[:100], y[99:100], 100)
    plt.plot(np.arange(100, 200), predictions)
    plt.show()



# print(generate_batches())
