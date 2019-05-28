# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# import keras as ks
from tensorflow.python.keras.layers import Input, GRU, Dense, Concatenate, TimeDistributed
from tensorflow.python.keras.models import Model

from layers.attention import AttentionLayer

# def build_seq2seq_attention_model(input_feature_amount, output_feature_amount, state_size, seq_len_in, seq_len_out):
#     """
#     Function to build the seq2seq model used.
#     :return: Encoder model, decoder model (used for predicting) and full model (used for training).
#     """
#     # Define model inputs for the encoder/decoder stack
#     encoder_input = ks.Input(shape=(seq_len_in, input_feature_amount), name="x_enc")
#     decoder_input = ks.Input(shape=(None, output_feature_amount), name="x_dec")
#
#     # Define the encoder GRU, which only has to return a state
#     encoder_gru = ks.layers.GRU(state_size, return_state=True, return_sequences=True)
#     encoder_intermediate_states, final_state = encoder_gru(encoder_input)
#
#
#
#
#
#
#     # Define the decoder GRU and the Dense layer that will transform sequences of size 20 vectors to
#     # a sequence of 1-long vectors of final predicted values
#     dec_gru = ks.layers.GRU(state_size, return_state=True, return_sequences=True)
#     # dec_dense2 = ks.layers.TimeDistributed(ks.layers.Dense(state_size, activation='relu'))
#     dec_dense = ks.layers.TimeDistributed(ks.layers.Dense(output_feature_amount, activation='linear'))
#
#     # Use these definitions to calculate the outputs of out encoder/decoder stack
#     dec_intermediates, _ = dec_gru(x_dec, initial_state=state)
#     # dec_intermediates = dec_dense2(dec_intermediates)
#     dec_outs = dec_dense(dec_intermediates)
#
#     # Define the encoder/decoder stack model
#     encdecmodel = ks.Model(inputs=[x_enc, x_dec], outputs=dec_outs)
#
#     # Define the encoder model
#     E = ks.Model(inputs=x_enc, outputs=state)
#
#     # Define a state_in model for the Decoder model (which will be used for prediction)
#     state_in = ks.Input(shape=(state_size,), name="state")
#
#     # Use the previously defined layers to calculate the new output value and state for the prediction model as well
#     dec_intermediate, new_state = dec_gru(x_dec, initial_state=state_in)
#     # dec_intermediate = dec_dense2(dec_intermediate)
#     dec_out = dec_dense(dec_intermediate)
#
#     # Define the decoder/prediction model
#     D = ks.Model(inputs=[x_dec, state_in], outputs=[dec_out, new_state])
#     return E, D, encdecmodel


# from models.seq2seq_attention.custom_recurrents import AttentionDecoder
#
#
# def build_seq2seq_attention_model(input_feature_amount, output_feature_amount, state_size, seq_len_in, seq_len_out):
#     """
#     Function to build the seq2seq model used.
#     :return: Encoder model, decoder model (used for predicting) and full model (used for training).
#     """
#     i = Input(shape=(seq_len_in, input_feature_amount), dtype='float32')
#     enc = Bidirectional(GRU(state_size, return_sequences=True), merge_mode='concat')(i)
#     dec = AttentionDecoder(state_size, seq_len_out)(enc)
#     model = Model(inputs=i, outputs=dec)
#     model.summary()


def build_seq2seq_attention_model(input_feature_amount, output_feature_amount, state_size, seq_len_in, seq_len_out):
    """
    Function to build the seq2seq model used.
    :return: Encoder model, decoder model (used for predicting) and full model (used for training).
    """
    # Define model inputs for the encoder/decoder stack
    x_enc = Input(shape=(seq_len_in, input_feature_amount), name="x_enc")
    x_dec = Input(shape=(seq_len_out, output_feature_amount), name="x_dec")

    # Define the encoder GRU, which only has to return a state
    encoder_gru = GRU(state_size, return_sequences=True, return_state=True, name="encoder_gru")
    encoder_out, encoder_state = encoder_gru(x_enc)

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

    # # Define the encoder model
    # E = ks.Model(inputs=x_enc, outputs=encoder_state)
    #
    # # Define a state_in model for the Decoder model (which will be used for prediction)
    # state_in = ks.Input(shape=(state_size,), name="state")
    #
    # # Use the previously defined layers to calculate the new output value and state for the prediction model as well
    # dec_intermediates, new_state = decoder_gru(x_dec, initial_state=encoder_state)
    #
    # # Define the attention layer
    # attn_layer = AttentionLayer(name="attention_layer")
    # attn_out, attn_states = attn_layer([encoder_out, dec_intermediates])
    #
    # # Concatenate decoder and attn out
    # decoder_concat_input = ks.layers.Concatenate(axis=-1, name='concat_layer')([dec_intermediates, attn_out])
    #
    # # Define the softmax layer
    # dense = ks.layers.Dense(state_size, activation='softmax', name='softmax_layer')
    # dense_time = ks.layers.TimeDistributed(dense, name='time_distributed_layer')
    # decoder_pred = dense_time(decoder_concat_input)
    #
    # print(x_dec)
    # print(state_in)
    # print(decoder_pred)
    # print(new_state)
    # D = ks.Model(inputs=[x_dec, state_in], outputs=[decoder_pred, new_state])

    # return E, D, encdecmodel

    # # Define the encoder/decoder stack model
    # encdecmodel = ks.Model(inputs=[x_enc, x_dec], outputs=decoder_out)
    #
    # # Define the encoder model
    # E = ks.Model(inputs=x_enc, outputs=encoder_state)
    #
    # # Define a state_in model for the Decoder model (which will be used for prediction)
    # state_in = ks.Input(shape=(state_size,), name="state")
    #
    # # Use the previously defined layers to calculate the new output value and state for the prediction model as well
    # dec_intermediate, new_state = dec_gru(x_dec, initial_state=state_in)
    # # dec_intermediate = dec_dense2(dec_intermediate)
    # dec_out = dec_dense(dec_intermediate)
    #
    # # Define the decoder/prediction model
    # D = ks.Model(inputs=[x_dec, state_in], outputs=[dec_out, new_state])
    # return E, D, encdecmodel









