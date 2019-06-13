import metrics
from models.ann import Ann
from models.seq2seq import Seq2Seq
from models.seq2seq_1dconv import Seq2SeqConv
from models.seq2seq_1dconv_attention import Seq2SeqConvAttention
from models.seq2seq_attention import Seq2SeqAttention
from utils import load_data, denormalize

import matplotlib.pyplot as plt
import numpy as np

# Load data
data_dict = load_data(
    "/home/mauk/Workspace/energy_prediction/data/prepared/aggregated_1415/aggregated_input_data-f83-ak75-b121.pkl")

batch_size = 128
state_size = 36
input_feature_amount = 83
output_feature_amount = 1
seq_len_in = 96
seq_len_out = 96
plot_time_steps_view = 96 * 2
steps_per_epoch = 15
epochs = 3
learning_rate = 0.00075
intermediates = 1
plot_loss = True

load_ann_weights_path = "/home/mauk/Workspace/energy_prediction/models/first_time_training_much data/ann-l0.00025-tl0.015-vl0.154-i96-o96-e2250-seq2seq.h5"
load_s2s_weights_path = "/home/mauk/Workspace/energy_prediction/models/first_time_training_much data/s2s-l0.00025-ss36-tl0.027-vl0.042-i96-o96-e2250-seq2seq.h5"
load_s2s_1dconv_weights_path = "/home/mauk/Workspace/energy_prediction/models/first_time_training_much data/s2s1dc-l0.00025-ss36-tl0.026-vl0.058-i96-o96-e2250-seq2seq.h5"
load_s2s_attention_weights_path = "/home/mauk/Workspace/energy_prediction/models/first_time_training_much data/as2s-l0.00025-ss36-tl0.025-vl0.047-i96-o96-e2250-seq2seq.h5"
load_s2s_1dconv_attention_weights_path = "/home/mauk/Workspace/energy_prediction/models/first_time_training_much data/as2s1dc-l0.00025-ss36-tl0.020-vl0.038-i96-o96-e2250-seq2seq.h5"


def train_models(models):
    """
    Train the given models
    :param models: The models
    """
    for model in models:
        model.train()


def plot_random_sample(models):
    """
    Plot a random sample with all the models predictions
    :param models: The models
    :return:
    """
    predict_x_batches, predict_y_batches, predict_y_batches_prev = seq2seq.create_validation_sample()
    predictions = {}

    for model in models:
        prediction = model.predict(predict_x_batches[0], predict_x_batches[1], predict_y_batches, predict_y_batches_prev)
        predictions[model.name] = denormalize(prediction, data_dict['output_std'], data_dict['output_mean'])

    normalized_ys = np.concatenate([predict_y_batches_prev, predict_y_batches[0]])[-plot_time_steps_view:]

    ys = denormalize(normalized_ys, data_dict['output_std'], data_dict['output_mean'])

    plt.plot(range(0, plot_time_steps_view), ys, label="real")

    for model_name in predictions.keys():
        plt.plot(range(plot_time_steps_view - seq_len_out, plot_time_steps_view), predictions[model_name],
                 label=model_name + " prediction")
    plt.legend()
    plt.title(label="predictions")
    plt.show()


def calculate_eval_loss_models(models):
    """
    Calculate the evaluation loss of all the models
    :param models: The models
    :return: The losses, as a dict, mapping the model to the loss
    """
    predict_x_batches, predict_y_batches = seq2seq.generate_validation_data()
    losses = {}

    for model in models:
        if "attention" in model.name:
            from tensorflow.python.keras.optimizers import SGD
            model.model.compile(SGD(1), metrics.root_mean_squared_error)
        else:
            from keras.optimizers import SGD
            model.model.compile(SGD(1), metrics.root_mean_squared_error)
        model_loss = model.model.evaluate(predict_x_batches, predict_y_batches, batch_size=len(predict_y_batches))
        losses[model] = model_loss

    return losses


def calculate_nrmse_models(models):
    """
    Calculate the NRMSE for all the models (max-min)
    :param models: The models
    :return: The calculated NRMSEs
    """
    losses = calculate_eval_loss_models(models)
    reals = data_dict['normalized_output_data']
    max_val = np.amax(reals)
    min_val = np.amin(reals)
    mean_val = np.mean(reals)
    nrmses = {}
    for model in losses.keys():
        nrmses[model] = losses[model] / (max_val - min_val)

    return nrmses


def print_nrmse_models(models):
    """
    Print the NMRSE of the models
    :param models: The models
    """
    nrmses = calculate_nrmse_models(models)
    print("=== NRMSE MODELS ===")
    for model in nrmses.keys():
        print(model.name + " nrmse: {0:.2f}%".format(nrmses[model] * 100))


if __name__ == "__main__":
    # Init models
    models = []

    seq2seq = Seq2Seq(data_dict=data_dict,
                      batch_size=batch_size,
                      state_size=state_size,
                      input_feature_amount=input_feature_amount,
                      output_feature_amount=output_feature_amount,
                      seq_len_in=seq_len_in,
                      seq_len_out=seq_len_out,
                      plot_time_steps_view=plot_time_steps_view,
                      steps_per_epoch=steps_per_epoch,
                      epochs=epochs,
                      learning_rate=learning_rate,
                      intermediates=intermediates,
                      plot_loss=plot_loss,
                      load_weights_path=load_s2s_weights_path
                      )

    seq2seq_1dconv = Seq2SeqConv(data_dict=data_dict,
                                 batch_size=batch_size,
                                 state_size=state_size,
                                 input_feature_amount=input_feature_amount,
                                 output_feature_amount=output_feature_amount,
                                 seq_len_in=seq_len_in,
                                 seq_len_out=seq_len_out,
                                 plot_time_steps_view=plot_time_steps_view,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 learning_rate=learning_rate,
                                 intermediates=intermediates,
                                 plot_loss=plot_loss,
                                 load_weights_path=load_s2s_1dconv_weights_path
                                 )

    ann = Ann(data_dict=data_dict,
              batch_size=batch_size,
              state_size=state_size,
              input_feature_amount=input_feature_amount,
              output_feature_amount=output_feature_amount,
              seq_len_in=seq_len_in,
              seq_len_out=seq_len_out,
              plot_time_steps_view=plot_time_steps_view,
              steps_per_epoch=steps_per_epoch,
              epochs=epochs,
              learning_rate=learning_rate,
              intermediates=intermediates,
              plot_loss=plot_loss,
              load_weights_path=load_ann_weights_path
              )

    seq2seq_attention = Seq2SeqAttention(data_dict=data_dict,
                                         batch_size=batch_size,
                                         state_size=state_size,
                                         input_feature_amount=input_feature_amount,
                                         output_feature_amount=output_feature_amount,
                                         seq_len_in=seq_len_in,
                                         seq_len_out=seq_len_out,
                                         plot_time_steps_view=plot_time_steps_view,
                                         steps_per_epoch=steps_per_epoch,
                                         epochs=epochs,
                                         learning_rate=learning_rate,
                                         intermediates=intermediates,
                                         plot_loss=plot_loss,
                                         load_weights_path=load_s2s_attention_weights_path
                                         )

    seq2seq_1dconv_attention = Seq2SeqConvAttention(data_dict=data_dict,
                                                    batch_size=batch_size,
                                                    state_size=state_size,
                                                    input_feature_amount=input_feature_amount,
                                                    output_feature_amount=output_feature_amount,
                                                    seq_len_in=seq_len_in,
                                                    seq_len_out=seq_len_out,
                                                    plot_time_steps_view=plot_time_steps_view,
                                                    steps_per_epoch=steps_per_epoch,
                                                    epochs=epochs,
                                                    learning_rate=learning_rate,
                                                    intermediates=intermediates,
                                                    plot_loss=plot_loss,
                                                    load_weights_path=load_s2s_1dconv_attention_weights_path
                                                    )

    models.append(seq2seq)
    models.append(seq2seq_1dconv)
    models.append(ann)
    models.append(seq2seq_attention)
    models.append(seq2seq_1dconv_attention)

    predict_x_batches, predict_y_batches, predict_y_batches_prev = seq2seq.create_validation_sample()

    print_nrmse_models(models)
