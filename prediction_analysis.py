import os
import pickle

import metrics
from models.ann import Ann
from models.seq2seq import Seq2Seq
from models.seq2seq_1dconv import Seq2SeqConv
from models.seq2seq_1dconv_attention import Seq2SeqConvAttention
from models.seq2seq_attention import Seq2SeqAttention
from utils import load_data, denormalize

import matplotlib.pyplot as plt
import numpy as np

agg_levels = [1, 25, 50, 75]
start_point_loss_graph = 25

batch_size = 64
state_size = 32
input_feature_amount = 83
output_feature_amount = 1
seq_len_in = 96
seq_len_out = 96
plot_time_steps_view = 96 * 2
steps_per_epoch = 10
epochs = 40
learning_rate = 0.00075
intermediates = 1
agg_level = 1
plot_loss = True

# Load data
data_dict = load_data(
    "/home/mauk/Workspace/energy_prediction/data/prepared/aggregated_1415/aggregated_input_data-f83-ak{}-b121.pkl".format(agg_level))

load_weights = False
if load_weights:
    load_ann_weights_path = "ann-ss{}-agg{}-best_weights.h5".format(state_size, agg_level)
    load_s2s_weights_path = "seq2seq-ss{}-agg{}-best_weights.h5".format(state_size, agg_level)
    load_s2s_1dconv_weights_path = "seq2seq_1dconv-ss{}-agg{}-best_weights.h5".format(state_size, agg_level)
    load_s2s_attention_weights_path = "seq2seq_attention-ss{}-agg{}-best_weights.h5".format(state_size, agg_level)
    load_s2s_1dconv_attention_weights_path = "seq2seq_1dconv_attention-ss{}-agg{}-best_weights.h5".format(state_size, agg_level)
else:
    load_ann_weights_path = None
    load_s2s_weights_path = None
    load_s2s_1dconv_weights_path = None
    load_s2s_attention_weights_path = None
    load_s2s_1dconv_attention_weights_path = None


def train_models(models):
    """
    Train the given models
    :param models: The models
    """
    for model in models:
        model.train()


def load_losses(path_to_history_folder):
    """
    Load all loss histories from a folder into a dict
    :param path_to_history_folder: Path to histories
    :return: Losses dict
    """
    losses = []
    for filename in os.listdir(path_to_history_folder):
        if "history" in filename:
            history_data_pkl = open(os.path.join(path_to_history_folder, filename), "rb")
            history_data = pickle.load(history_data_pkl)
            losses.append(history_data)
    return losses


def plot_loss_graph_single_model(losses_dict):
    """
    Plot the loss of a single model, given a dictionary with train_losses and validation_losses
    :param losses_dict: The losses dict
    """
    train_losses = losses_dict['train_losses']
    val_losses = losses_dict['val_losses']

    plt.plot(train_losses, label="training_loss")
    plt.plot(val_losses, label="validation_loss")
    plt.legend()
    plt.title(label=losses_dict['name'] + " loss")
    plt.show()


def plot_loss_graph_validation(losses_dict_list, agg_lvl=None, plot_ann=False):
    """
    Plot all validation losses from a losses dict
    :param losses_dict_list: Dict containing validation losses
    """
    for loss_dict in losses_dict_list:
        if agg_lvl:
            if str(agg_lvl) in loss_dict['name'].split("-")[1]:
                if "ann" in loss_dict['name'] and not plot_ann:
                    continue
                plt.plot(loss_dict['val_losses'][start_point_loss_graph:], label=loss_dict['name'])
        else:
            if "ann" in loss_dict['name'] and not plot_ann:
                continue
            plt.plot(loss_dict['val_losses'][start_point_loss_graph:], label=loss_dict['name'])
    plt.legend()
    plt.title(label="validation losses")
    plt.show()


def plot_loss_graph_training(losses_dict_list, agg_lvl=None, plot_ann=False):
    """
    Plot all training losses from a losses dict
    :param losses_dict_list: Dict containing validation losses
    """
    for loss_dict in losses_dict_list:
        if agg_lvl:
            if str(agg_lvl) in loss_dict['name'].split("-")[1]:
                if "ann" in loss_dict['name'] and not plot_ann:
                    continue
                plt.plot(loss_dict['train_losses'][start_point_loss_graph:], label=loss_dict['name'])
        else:
            if "ann" in loss_dict['name'] and not plot_ann:
                continue
            plt.plot(loss_dict['train_losses'][start_point_loss_graph:], label=loss_dict['name'])
    plt.legend()
    plt.title(label="training losses")
    plt.show()


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
    :return: The losses_dict, as a dict, mapping the model to the loss
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

    seq2seq = Seq2Seq(name="seq2seq",
                      data_dict=data_dict,
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
                      load_weights_path=load_s2s_weights_path,
                      agg_level=agg_level
                      )

    seq2seq_1dconv = Seq2SeqConv(name="seq2seq_1dconv",
                                 data_dict=data_dict,
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
                                 load_weights_path=load_s2s_1dconv_weights_path,
                                 agg_level=agg_level
                                 )

    ann = Ann(name="ann",
              data_dict=data_dict,
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
              load_weights_path=load_ann_weights_path,
              agg_level=agg_level
              )

    seq2seq_attention = Seq2SeqAttention(name="seq2seq_attention",
                                         data_dict=data_dict,
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
                                         load_weights_path=load_s2s_attention_weights_path,
                                         agg_level=agg_level
                                         )

    seq2seq_1dconv_attention = Seq2SeqConvAttention(name="seq2seq_1dconv_attention",
                                                    data_dict=data_dict,
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
                                                    load_weights_path=load_s2s_1dconv_attention_weights_path,
                                                    agg_level=agg_level
                                                    )

    models.append(seq2seq_attention)
    models.append(seq2seq_1dconv_attention)
    models.append(seq2seq)
    models.append(seq2seq_1dconv)
    models.append(ann)

    for model in models:
        model.model.summary()

    # train_models(models)

    # predict_x_batches, predict_y_batches, predict_y_batches_prev = seq2seq.create_validation_sample()
    #
    # print_nrmse_models(models)

    # losses_dict = load_losses("/home/mauk/Workspace/energy_prediction/")

    # plot_random_sample(models)

    # for agg_lvl in agg_levels:
    #     plot_loss_graph_validation(losses_dict, agg_lvl=agg_lvl, plot_ann=False)
    #     plot_loss_graph_training(losses_dict, agg_lvl=agg_lvl, plot_ann=False)
