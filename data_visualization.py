import os
import random

import matplotlib.pyplot as plt
import pandas as pd

day = 96
week = day * 7
month = day * 30
year = day * 365

size_data = 69986


def visualize_data(path_to_data):
    df = pd.read_csv(path_to_data)
    sp = random.randint(0, len(df.index) - 1000)  # Add arbitrary large number so it doesn't break
    df = df[sp:sp+month]

    df = df.loc[:, (df != 0).any(axis=0)]

    for column in df:
        try:
            temp_df = df[column]
            plot = temp_df.plot(title=column)
            plot.plot()
            plt.show()
        except TypeError:
            print("Warning, type error in " + column)


def visualize_column(path_to_data, column_name):
    df = pd.read_csv(path_to_data)
    sp = random.randint(0, len(df.index) - 1000)  # Add arbitrary large number so it doesn't break
    df = df[sp:sp+week]

    df = df.loc[:, (df != 0).any(axis=0)]

    try:
        temp_df = df[column_name]
        plot = temp_df.plot(title=column_name)
        plot.plot()
        plt.show()
    except TypeError:
        print("Warning, type error in " + column_name)


def visualize_column_from_multiple(path_to_data_folder, column_name, size_data):
    sp = random.randint(0, size_data - 1000)  # Add arbitrary large number so it doesn't break
    for filename in os.listdir(path_to_data_folder):
        if ".csv" in filename:
            print("Processing", filename)
            df = pd.read_csv(os.path.join(path_to_data_folder, filename))
            df = df[sp:sp+week]

            df = df.loc[:, (df != 0).any(axis=0)]

            try:
                temp_df = df[column_name]
                plot = temp_df.plot(title=filename + " " + column_name)
                plot.plot()
                plt.show()
            except TypeError:
                print("Warning, type error in " + filename + ", " + column_name)


def visualize_column_on_interval(path_to_data, column_name):
    time_range = week
    df = pd.read_csv(path_to_data)
    sp = 300  # random.randint(0, len(df.index) - 1000)  # Add arbitrary large number so it doesn't break

    df = df.loc[:, (df != 0).any(axis=0)]

    while 0 <= sp < size_data - time_range:
        temp_df = df[sp:sp + time_range]
        temp_df = temp_df[column_name]
        plot = temp_df.plot(title=column_name + str(sp))
        plot.plot()
        plt.show()
        sp = sp + time_range


# visualize_column_from_multiple("/home/mauk/Workspace/energy_prediction/data/prepared/aggregated_1415/", "use", size_data=size_data)
# visualize_data("/home/mauk/Workspace/energy_prediction/data/prepared/aggregated_1415/p-agg-0.csv")
visualize_column("/home/mauk/Workspace/energy_prediction/data/prepared/aggregated_1415/p-agg-24.csv", "use")
# visualize_column("/home/mauk/Workspace/energy_prediction/data/raw/building_energy/1415/2242-building_data-1415.csv", "use")
# visualize_column_on_interval("/home/mauk/Workspace/energy_prediction/data/prepared/aggregated_1415/p-agg-35.csv", "use")
# visualize_column_from_multiple("/home/mauk/Workspace/energy_prediction/data/raw/building_energy/1415/", "use", size_data=size_data)

