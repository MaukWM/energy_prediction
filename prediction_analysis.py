from models.main import generate_validation_data, input_feature_amount, output_feature_amount, state_size, seq_len_in, \
    seq_len_out, generate_validation_sample, generate_training_sample, plot_last_time_steps_view, output_std, \
    output_mean
from models.seq2seq.seq2seq import build_seq2seq_model
from models.seq2seq.training import s2s_calculate_accuracy, s2s_predict
from models.seq2seq_attention.seq2seq_attention import build_seq2seq_attention_model
from models.seq2seq_attention.training import s2s_attention_predict, s2s_attention_calculate_accuracy

import numpy as np
import matplotlib.pyplot as plt

from utils import denormalize

models = []

s2s_attention_encoder, s2s_attention_decoder, s2s_attention_encdecmodel = build_seq2seq_attention_model(
    input_feature_amount=input_feature_amount,
    output_feature_amount=output_feature_amount,
    state_size=state_size, seq_len_in=seq_len_in,
    seq_len_out=seq_len_out)

s2s_attention_encdecmodel.load_weights(
    filepath="/home/mauk/Workspace/energy_prediction/models/seq2seq_attention/as2s-l0.00075-ss78-tl0.098-vl0.104-i192-o96-e300-seq2seq.h5")

s2s_encoder, s2s_decoder, s2s_encdecmodel = build_seq2seq_model(input_feature_amount=input_feature_amount,
                                                                output_feature_amount=output_feature_amount,
                                                                state_size=state_size)

s2s_encdecmodel.load_weights(
    filepath="/home/mauk/Workspace/energy_prediction/models/seq2seq/s2s-l0.00075-ss78-tl0.099-vl0.099-i192-o96-e300-seq2seq.h5")


def calculate_accuracy():
    pass


# TODO: Change to give list of models and just run .predict instead of hardcoding
def predict(predict_x_batches, actual_output, prev_output):
    s2s_norm_attention_prediction = s2s_attention_predict(s2s_attention_encoder, s2s_attention_decoder, predict_x_batches[0], predict_x_batches[1],
                          predict_y_batches, predict_y_batches_prev)

    s2s_norm_prediction = s2s_predict(s2s_encoder, s2s_decoder, predict_x_batches[0], predict_x_batches[1], predict_y_batches,
                predict_y_batches_prev)

    # Concat the normalized_ys so we get a smooth line for the normalized_ys
    normalized_ys = np.concatenate([prev_output, actual_output[0]])[-plot_last_time_steps_view:]

    ys = denormalize(normalized_ys, output_std, output_mean)
    s2s_attention_prediction = denormalize(s2s_norm_attention_prediction, output_std, output_mean)
    s2s_prediction = denormalize(s2s_norm_prediction, output_std, output_mean)

    # Plot them
    plt.plot(range(0, plot_last_time_steps_view), ys, label="real")
    plt.plot(range(plot_last_time_steps_view - seq_len_out, plot_last_time_steps_view), s2s_attention_prediction, label="s2s_attention_prediction")
    plt.plot(range(plot_last_time_steps_view - seq_len_out, plot_last_time_steps_view), s2s_prediction,
             label="s2s_prediction")
    plt.legend()
    plt.title(label="predictions")
    plt.show()



# TODO: Make class for each method so this does not have to be so ugly
if __name__ == "__main__":
    predict_x_batches, predict_y_batches, predict_y_batches_prev = generate_validation_sample()

    # Predict for s2s_attention
    s2s_attention_calculate_accuracy(predict_x_batches, predict_y_batches, predict_y_batches_prev,
                                     s2s_attention_encdecmodel, s2s_attention_encoder, s2s_attention_decoder)

    # s2s_prediction = s2s_attention_predict(s2s_attention_encoder, s2s_attention_decoder, predict_x_batches[0], predict_x_batches[1],
    #                       predict_y_batches, predict_y_batches_prev)

    # Predict for s2s
    s2s_calculate_accuracy(predict_x_batches, predict_y_batches, predict_y_batches_prev, s2s_encdecmodel,
                           s2s_encoder, s2s_decoder)

    # s2s_predict(s2s_encoder, s2s_decoder, predict_x_batches[0], predict_x_batches[1], predict_y_batches,
    #             predict_y_batches_prev)

    predict(predict_x_batches, predict_y_batches, predict_y_batches_prev)

