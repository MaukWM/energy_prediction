import math
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
plot_time_steps_view = 96
steps_per_epoch = 10
epochs = 40
learning_rate = 0.00075
intermediates = 1
agg_level = 75
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


def plot_random_day_sample(models):
    """
    Plot a random sample, from the start until the end of the day, with all the models predictions
    :param models: The models
    """
    predict_x_batches, predict_y_batches, predict_y_batches_prev = seq2seq.create_validation_sample(is_start_of_day=True)
    predictions = {}

    for model in models:
        prediction = model.predict(predict_x_batches[0], predict_x_batches[1], predict_y_batches,
                                   predict_y_batches_prev)
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
        print("model_loss", model_loss)
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


def predict_all_validation_data(models, save=True):
    """
    Make predictions over the validation data, saving the predicted value and actual value
    :param models: The models
    :return Dict containing the predicted and actual values of the all datapoints in the validation data
    """
    slicing_point = 1500
    test_xe_batches, test_xd_batches, test_y_batches, test_y_batches_prev = models[0].create_validation_data_with_prev_y_steps(slice_point=slicing_point)

    values_dict = dict()

    # Initialize dict
    for model in models:
        values_dict[model.name] = []

    print("Total datapoints to process with slicing point {}:".format(slicing_point), len(test_xe_batches))
    update_point = int(len(test_xe_batches) / 10)
    for i in range(len(test_xe_batches)):
        if i % update_point == 0:
            print("Processing datapoint", i)
        # Reshape into batch
        test_xe_batches_inf = np.reshape(test_xe_batches[i], newshape=(1, seq_len_in, input_feature_amount))
        test_xd_batches_inf = np.reshape(test_xd_batches[i], newshape=(1, seq_len_in, output_feature_amount))
        test_y_batches_inf = np.reshape(test_y_batches[i], newshape=(1, seq_len_out, output_feature_amount))

        normalized_ys = test_y_batches[i]
        ys = denormalize(normalized_ys, data_dict['output_std'], data_dict['output_mean'])

        for model in models:
            prediction = model.predict(test_xe_batches_inf, test_xd_batches_inf, test_y_batches_inf,
                                       test_y_batches_prev[i], plot=False)
            denormalized_prediction = denormalize(prediction, data_dict['output_std'], data_dict['output_mean'])

            result = (denormalized_prediction, ys)
            values_dict[model.name].append(result)

    if save:
        out_file = open("predicted_and_actuals-agg{}-sp{}.pkl".format(agg_level, slicing_point), "wb")
        pickle.dump(values_dict, out_file)

    return values_dict


def calculate_accuracy_per_time_step(models, plot=True, save=True):
    """
    Plot the accuracy of each model for each timestep and plot it.
    :param models: The models
    :return Dict containing the average RMSE of each model for each timestep.
    """
    slicing_point = 1500
    test_xe_batches, test_xd_batches, test_y_batches, test_y_batches_prev = models[0].create_validation_data_with_prev_y_steps(slice_point=slicing_point)

    rmse_dict = {}

    # Initialize dict containing the rmses
    for model in models:
        rmse_dict[model.name] = []

    print("Total datapoints to process with slicing point {}:".format(slicing_point), len(test_xe_batches))
    update_point = int(len(test_xe_batches) / 10)
    for i in range(len(test_xe_batches)):
        if i % update_point == 0:
            print("Processing datapoint", i)
        # Reshape into batch
        test_xe_batches_inf = np.reshape(test_xe_batches[i], newshape=(1, seq_len_in, input_feature_amount))
        test_xd_batches_inf = np.reshape(test_xd_batches[i], newshape=(1, seq_len_in, output_feature_amount))
        test_y_batches_inf = np.reshape(test_y_batches[i], newshape=(1, seq_len_out, output_feature_amount))

        normalized_ys = test_y_batches[i]
        ys = denormalize(normalized_ys, data_dict['output_std'], data_dict['output_mean'])

        for model in models:
            prediction = model.predict(test_xe_batches_inf, test_xd_batches_inf, test_y_batches_inf,
                                       test_y_batches_prev[i], plot=False)
            denormalized_prediction = denormalize(prediction, data_dict['output_std'], data_dict['output_mean'])

            rmse_result = []
            rmse_result.append(np.sqrt(np.square(ys - denormalized_prediction)))
            rmse_dict[model.name].append(rmse_result)

    for model in rmse_dict.keys():
        rmse_dict[model] = np.average(rmse_dict[model], axis=0)

    if save:
        out_file = open("avg_rmse_timesteps-agg{}-sp{}.pkl".format(agg_level, slicing_point), "wb")
        pickle.dump(rmse_dict, out_file)

    if plot:
        plt.title("RMSE for {}".format(agg_level))
        plt.xlabel("Timestep (15 min)")
        plt.ylabel("Average RMSE")

        for model in rmse_dict.keys():
            plt.plot(rmse_dict[model][0], label=model)

        plt.legend()
        plt.show()

    return rmse_dict


def plot_accuracy_per_time_step(path_to_rmse_dict):
    """
    Plot the RMSE per timestep given the path to the dict location
    :param path_to_rmse_dict: Path to RMSE dict
    """
    data_tmp = open(path_to_rmse_dict, "rb")
    rmse_dict = pickle.load(data_tmp)

    plt.title("RMSE for aggregation level {}".format(agg_level))
    plt.xlabel("Timestep (15 min)")
    plt.ylabel("Average RMSE")

    for model in rmse_dict.keys():
        plt.plot(rmse_dict[model][0], label=model)

    plt.legend()
    plt.show()


def print_accuracy_measures(predictions, actuals):
    """
    Print the results of all accuracy measures, given the predictions and actuals
    :param predictions: Predictions
    :param actuals: Actuals
    """
    # Stack the lists to make them numpy arrays
    predictions = np.stack(predictions)
    actuals = np.stack(actuals)

    # Get total amount of observations
    n = np.size(predictions)

    y_max = np.amax(actuals)
    y_min = np.amin(actuals)

    # Mean error
    me = np.sum(predictions - actuals) / n
    print("ME: {0:.2f}".format(me))

    temp = np.sum(np.square(predictions - actuals), axis=1)
    temp = np.reshape(temp, newshape=len(temp))

    plt.plot(np.cumsum(temp))
    plt.show()

    # Root mean squared error
    rmse = math.sqrt(np.sum(np.square(predictions - actuals)) / n)
    print("RMSE: {0:.2f}".format(rmse))

    # Mean absolute error
    mae = np.sum(abs(predictions - actuals)) / n
    print("MAE: {0:.2f}".format(mae))

    # Normalized root mean squared error
    nrmse = rmse / (y_max - y_min) * 100
    print("NRMSE: {0:.2f}".format(nrmse))


def analyze_predicted_and_actuals(path_to_data_folder):
    """
    Analyze predicted and actual values by calculating the values of some metrics
    :param path_to_data_folder: Path to data folder
    """
    pred_act_dict = dict()
    for filename in os.listdir(path_to_data_folder):
        if "predicted_and_actuals" in filename:
            print("Loading", filename)
            data_tmp = open(os.path.join(path_to_data_folder, filename), "rb")
            # Ugly way to extract agg_level from filename
            agg_level = filename.split("-")[1].split(".")[0][3:]
            data = pickle.load(data_tmp)
            pred_act_dict[agg_level] = data

    for pred_act_agg_level in pred_act_dict.keys():
        for model in pred_act_dict[pred_act_agg_level]:
            predictions = []
            actuals = []
            for data_point in pred_act_dict[pred_act_agg_level][model]:
                predictions.append(data_point[0])
                actuals.append(data_point[1])
            print("Accuracy measures for {} with {} aggregation".format(model, pred_act_agg_level))
            print_accuracy_measures(predictions, actuals)


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

    # plot_accuracy_per_time_step("/home/mauk/Workspace/energy_prediction/avg_rmse_timesteps-agg1.pkl")
    # plot_accuracy_per_time_step("/home/mauk/Workspace/energy_prediction/avg_rmse_timesteps-agg25.pkl")
    # plot_accuracy_per_time_step("/home/mauk/Workspace/energy_prediction/avg_rmse_timesteps-agg50.pkl")
    # plot_accuracy_per_time_step("/home/mauk/Workspace/energy_prediction/avg_rmse_timesteps-agg75.pkl")

    # calculate_accuracy_per_time_step(models)

    # predict_all_validation_data(models)
    
    analyze_predicted_and_actuals(path_to_data_folder="/home/mauk/Workspace/energy_prediction/")

    for model in models:
        model.model.summary()

    print_nrmse_models(models)
    #
    # losses_dict = load_losses("/home/mauk/Workspace/energy_prediction/")
    #
    # plot_random_sample(models)
    # plot_random_day_sample(models)
    #
    # for agg_lvl in agg_levels:
    #     plot_loss_graph_validation(losses_dict, agg_lvl=agg_lvl, plot_ann=True)
    #     plot_loss_graph_training(losses_dict, agg_lvl=agg_lvl, plot_ann=True)
