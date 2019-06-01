from tensorflow.python.keras.layers import Input, GRU, Dense, Concatenate, TimeDistributed, Conv1D
from tensorflow.python.keras.models import Model

from layers.attention import AttentionLayer


def build_seq2seq_1dconv_attention_model(input_feature_amount, output_feature_amount, state_size, seq_len_in, seq_len_out):
    """
    Function to build the seq2seq model used.
    :return: Encoder model, decoder model (used for predicting) and full model (used for training).
    """
    # Define model inputs for the encoder/decoder stack
    x_enc = Input(shape=(seq_len_in, input_feature_amount), name="x_enc")
    x_dec = Input(shape=(seq_len_out, output_feature_amount), name="x_dec")

    input_conv4 = Conv1D(filters=46, kernel_size=7, strides=2, activation='relu')(x_enc)
    input_conv3 = Conv1D(filters=46, kernel_size=5, strides=1, activation='relu')(input_conv4)
    input_conv2 = Conv1D(filters=46, kernel_size=5, strides=2, activation='relu')(input_conv3)
    input_conv1 = Conv1D(filters=46, kernel_size=3, strides=1, activation='relu')(input_conv2)
    input_conv = Conv1D(filters=46, kernel_size=3, strides=2, activation='relu', name="last_conv_layer")(input_conv1)

    # Define the encoder GRU, which only has to return a state
    encoder_gru = GRU(state_size, return_sequences=True, return_state=True, name="encoder_gru")
    encoder_out, encoder_state = encoder_gru(input_conv)

    # Decoder GRU
    decoder_gru = GRU(state_size, return_state=True, return_sequences=True,
                                  name="decoder_gru")
    # Use these definitions to calculate the outputs of out encoder/decoder stack
    dec_intermediates, decoder_state = decoder_gru(x_dec, initial_state=encoder_state)

    # Define the attention layer
    attn_layer = AttentionLayer(name="attention_layer")
    attn_out, attn_states = attn_layer([encoder_out, dec_intermediates])

    # Concatenate decoder and attn out
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([dec_intermediates, attn_out])

    # Define the dense layer
    dense = Dense(output_feature_amount, activation='linear', name='softmax_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    decoder_pred = dense_time(decoder_concat_input)

    # Define the encoder/decoder stack model
    encdecmodel = Model(inputs=[x_enc, x_dec], outputs=decoder_pred)

    # Define the separate encoder model for inferencing
    encoder_inf_inputs = Input(shape=(seq_len_in, input_feature_amount), name="encoder_inf_inputs")

    input_conv4 = Conv1D(filters=46, kernel_size=7, strides=2, activation='relu')(encoder_inf_inputs)
    input_conv3 = Conv1D(filters=46, kernel_size=5, strides=1, activation='relu')(input_conv4)
    input_conv2 = Conv1D(filters=46, kernel_size=5, strides=2, activation='relu')(input_conv3)
    input_conv1 = Conv1D(filters=46, kernel_size=3, strides=1, activation='relu')(input_conv2)
    input_conv = Conv1D(filters=46, kernel_size=3, strides=2, activation='relu', name="last_conv_layer")(input_conv1)

    encoder_inf_out, encoder_inf_state = encoder_gru(input_conv)
    encoder_model = Model(inputs=encoder_inf_inputs, outputs=[encoder_inf_out, encoder_inf_state])

    # Define the separate encoder model for inferencing
    decoder_inf_inputs = Input(shape=(1, output_feature_amount), name="decoder_inputs")
    encoder_inf_states = Input(shape=(20, state_size), name="decoder_inf_states")  # TODO: Unhardcode the 20, get output shape of last conv layer!
    decoder_init_state = Input(shape=(state_size,), name="decoder_init")

    decoder_inf_out, decoder_inf_state = decoder_gru(decoder_inf_inputs, initial_state=decoder_init_state)
    attn_inf_out, attn_inf_states = attn_layer([encoder_inf_states, decoder_inf_out])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_inf_out, attn_inf_out])
    decoder_inf_pred = TimeDistributed(dense)(decoder_inf_concat)
    decoder_model = Model(inputs=[encoder_inf_states, decoder_init_state, decoder_inf_inputs],
                          outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_state])

    return encoder_model, decoder_model, encdecmodel

