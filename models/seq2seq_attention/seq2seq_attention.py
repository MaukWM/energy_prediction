from keras.layers import GaussianNoise
from tensorflow.python.keras.layers import Input, GRU, Dense, Concatenate, TimeDistributed
from tensorflow.python.keras.models import Model

from layers.attention import AttentionLayer


def build_seq2seq_attention_model(input_feature_amount, output_feature_amount, state_size, seq_len_in, seq_len_out, use_noise=True):
    """
    Function to build the seq2seq model used.
    :return: Encoder model, decoder model (used for predicting) and full model (used for training).
    """
    # Define model inputs for the encoder/decoder stack
    x_enc = Input(shape=(seq_len_in, input_feature_amount), name="x_enc")
    x_dec = Input(shape=(seq_len_out, output_feature_amount), name="x_dec")

    # Add noise
    if use_noise:
        x_dec_t = GaussianNoise(0.2)(x_dec)
    else:
        x_dec_t = x_dec

    # Define the encoder GRU, which only has to return a state
    encoder_gru = GRU(state_size, return_sequences=True, return_state=True, name="encoder_gru")
    encoder_out, encoder_state = encoder_gru(x_enc)

    # Decoder GRU
    decoder_gru = GRU(state_size, return_state=True, return_sequences=True,
                                name="decoder_gru")
    # Use these definitions to calculate the outputs of out encoder/decoder stack
    dec_intermediates, decoder_state = decoder_gru(x_dec_t, initial_state=encoder_state)

    # Define the attention layer
    attn_layer = AttentionLayer(name="attention_layer")
    attn_out, attn_states = attn_layer([encoder_out, dec_intermediates])

    # Concatenate decoder and attn out
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([dec_intermediates, attn_out])

    # Define the dense layer
    dense = Dense(output_feature_amount, activation='linear', name='output_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    decoder_pred = dense_time(decoder_concat_input)

    # Define the encoder/decoder stack model
    encdecmodel = Model(inputs=[x_enc, x_dec], outputs=decoder_pred)

    # Define the separate encoder model for inferencing
    encoder_inf_inputs = Input(shape=(seq_len_in, input_feature_amount), name="encoder_inf_inputs")
    encoder_inf_out, encoder_inf_state = encoder_gru(encoder_inf_inputs)
    encoder_model = Model(inputs=encoder_inf_inputs, outputs=[encoder_inf_out, encoder_inf_state])

    # Define the separate encoder model for inferencing
    decoder_inf_inputs = Input(shape=(1, output_feature_amount), name="decoder_inputs")
    encoder_inf_states = Input(shape=(seq_len_in, state_size), name="encoder_inf_states")
    decoder_init_state = Input(shape=(state_size,), name="decoder_init")

    decoder_inf_out, decoder_inf_state = decoder_gru(decoder_inf_inputs, initial_state=decoder_init_state)
    attn_inf_out, attn_inf_states = attn_layer([encoder_inf_states, decoder_inf_out])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_inf_out, attn_inf_out])
    decoder_inf_pred = TimeDistributed(dense)(decoder_inf_concat)
    decoder_model = Model(inputs=[encoder_inf_states, decoder_init_state, decoder_inf_inputs],
                          outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_state])

    return encoder_model, decoder_model, encdecmodel
