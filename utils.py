import os
import pickle
import matplotlib.pyplot as plt


def load_data(pkl_path=None):
    """
    Returns plain input and output data without any info on output normalization
    :param pkl_path: Path to pkl file
    :return: Normalized input data and output data
    """
    if pkl_path:
        input_data = open(pkl_path, "rb")
        return pickle.load(input_data)
    start_dir = "../.."
    pkl_data = "input_data.pkl"
    for root, dirs, files in os.walk(start_dir):
        if pkl_data in files:
            input_data = open(os.path.join(root, pkl_data), "rb")
            return pickle.load(input_data)
    print("ERROR: Pickle file was not found!")


# TODO: Remove redundant code
def load_data_normalized(pkl_path=None):
    """
    Returns input and output data with info on output normalization to denormalize
    :param pkl_path: Path to pkl file
    :return: Dict with normalized input data and output data and factors to denormalize output
    """
    if pkl_path:
        input_data = open(pkl_path, "rb")
        return pickle.load(input_data)
    start_dir = "../.."
    pkl_data = "input_data.pkl"
    for root, dirs, files in os.walk(start_dir):
        if pkl_data in files:
            input_data = open(os.path.join(root, pkl_data), "rb")
            return pickle.load(input_data)
    print("ERROR: Pickle file was not found!")


def denormalize(to_denormalize, std, mean):
    """
    Denormalize an array
    :param to_denormalize: Array to denormalize
    :param std: standard deviation
    :param mean: mean
    :return: Denormalized array
    """
    return (to_denormalize * std) + mean


def plot_attention_weights(attention_weights):
    """
    Plots attention weights
    :param attention_weights: sequence of attention weights: [seq_len_out, seq_len_in]
    :return:
    """
    plt.matshow(attention_weights)

    plt.show()

    attention_weights.transpose()
