import keras as ks
from keras import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape


def build_ann_model(input_feature_amount, output_feature_amount, seq_len_in, seq_len_out):
    """
    Function to build the seq2seq model used.
    :return: Encoder model, decoder model (used for predicting) and full model (used for training).
    """
    # model = Sequential([
    #     Dense(64, input_shape=(input_feature_amount * seq_len_in,)),
    #     Activation('relu'),
    #     Dense(output_feature_amount * seq_len_out),
    #     Activation('linear'),
    # ])

    input_sequence = ks.Input(shape=(seq_len_in, input_feature_amount), name="input")
    not_used = ks.Input(shape=(None, output_feature_amount), name="not_used")

    input_flat = Flatten()(input_sequence)

    dense1 = ks.layers.Dense(2048, activation='relu')
    dense1_out = dense1(input_flat)

    dense2 = ks.layers.Dense(256, activation='relu')
    dense2_out = dense2(dense1_out)

    dense3 = ks.layers.Dense(seq_len_out * output_feature_amount, activation='linear')
    dense3_out = dense3(dense2_out)
    seq_out = Reshape((seq_len_out, output_feature_amount))(dense3_out)

    model = ks.Model(inputs=[input_sequence, not_used], outputs=seq_out)

    return model
