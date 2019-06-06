import random

import matplotlib.pyplot as plt
import pandas as pd

day = 96
week = day * 7
month = day * 30
year = day * 365


def visualize_day_data(path_to_data):
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



visualize_day_data("/home/mauk/Workspace/energy_prediction/data/cleaned/building_energy/tc-3192-building_data-1415.csv")