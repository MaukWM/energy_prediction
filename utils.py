import os
import pickle


def load_data():
    start_dir = "."
    pkl_data = "input_data.pkl"
    for root, dirs, files in os.walk(start_dir):
        if pkl_data in files:
            input_data = open(os.path.join(root, pkl_data), "rb")
            return pickle.load(input_data)
            # open("../data/prepared/input_data.pkl", "rb")