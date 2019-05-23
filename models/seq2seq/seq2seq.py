# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
import keras as ks


def build_seq2seq_model(input_feature_amount, output_feature_amount, state_size, use_noise=False):
    """
    Function to build the seq2seq model used.
    :return: Encoder model, decoder model (used for predicting) and full model (used for training).
    """
    # Define model inputs for the encoder/decoder stack
    x_enc = ks.Input(shape=(None, input_feature_amount), name="x_enc")
    x_dec = ks.Input(shape=(None, output_feature_amount), name="x_dec")

    # input_conv = ks.layers.Conv1D(filters=32, kernel_size=8, strides=4, activation='relu', input_shape=(seq_len_in, input_feature_amount))(x_enc)
    # input_conv_pool = ks.layers.MaxPooling1D(pool_size=2)(input_conv)

    if use_noise:
        x_dec_t = ks.layers.GaussianNoise(0.2)(x_dec)
    else:
        x_dec_t = x_dec

    # Define the encoder GRU, which only has to return a state
    _, state = ks.layers.GRU(state_size, return_state=True)(x_enc)  # (input_conv)

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






