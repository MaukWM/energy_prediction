import os
import pickle
import matplotlib.pyplot as plt


def load_data(pkl_path=None):
    if pkl_path:
        input_data = open(pkl_path, "rb")
        return pickle.load(input_data)
    start_dir = "../.."
    pkl_data = "input_data-f84-0206.pkl"
    for root, dirs, files in os.walk(start_dir):
        if pkl_data in files:
            input_data = open(os.path.join(root, pkl_data), "rb")
            return pickle.load(input_data)
            # open("../data/prepared/input_data-f84-0206.pkl", "rb")
    print("ERROR: Pickle file was not found!")


def plot_attention_weights(attention_weights):
    """
    Plots attention weights
    :param attention_weights: sequence of attention weights: [seq_len_out, seq_len_in]
    :return:
    """
    plt.matshow(attention_weights)

    plt.show()

    attention_weights.transpose()
