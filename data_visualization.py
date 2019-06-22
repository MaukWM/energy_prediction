import datetime
import os
import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.dates import HourLocator, DateFormatter

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


def visualize_column_24hour(path_to_data, column_name):
    df = pd.read_csv(path_to_data)
    start_points = np.arange(0, len(df.index), 96)
    sp = start_points[random.randint(0, len(start_points) - 1)]
    df = df[sp:sp+day+1]

    df = df.loc[:, (df != 0).any(axis=0)]

    customdate = datetime.datetime(2016, 1, 1, 0, 0)

    y = df[column_name]
    x = [customdate + datetime.timedelta(minutes=15*i) for i in range(len(y))]

    ax = plt.subplot()

    # removing top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # adds major gridlines
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    ax.plot(x, y, color='black', linewidth=0.6)
    ax.xaxis.set_major_locator(HourLocator(interval=3))
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.xaxis.set_ticks([customdate + datetime.timedelta(hours=i*3) for i in range(int(24 / 3) + 1)])
    ax.set_xlabel("Time (15-min interval)")
    ax.set_ylabel("Energy use [kWh]")

    # beautify the x-labels
    # plt.gcf().autofmt_xdate()

    plt.show()


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
# visualize_column("/home/mauk/Workspace/energy_prediction/data/prepared/aggregated_1415/p-agg-24.csv", "use")
# visualize_column_on_interval("/home/mauk/Workspace/energy_prediction/data/prepared/aggregated_1415/p-agg-24.csv", "use")
visualize_column_24hour("/home/mauk/Workspace/energy_prediction/data/prepared/aggregated_1415/agg121-p/p-agg-0.csv", "use")
# visualize_column("/home/mauk/Workspace/energy_prediction/data/raw/building_energy/1415/2242-building_data-1415.csv", "use")
# visualize_column_on_interval("/home/mauk/Workspace/energy_prediction/data/prepared/aggregated_1415/p-agg-35.csv", "use")
# visualize_column_from_multiple("/home/mauk/Workspace/energy_prediction/data/raw/building_energy/1415/", "use", size_data=size_data)

