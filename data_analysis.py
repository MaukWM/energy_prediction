import os

import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.width', 1000)


def agg_data_and_visualize():
    """
    Aggregate all raw data into one week and display.
    Also sum and agg all data from all buildings for one week
    :return:
    """
    # Agg all data into one week
    path_to_data_folder = "/home/mauk/Workspace/energy_prediction/data/raw/building_energy/1415/"

    dfs = []
    for filename in os.listdir(path_to_data_folder):
        if ".csv" in filename:
            print("Processing", filename)
            df = pd.read_csv(os.path.join(path_to_data_folder, filename))

            df["local_15min"] = pd.to_datetime(df["local_15min"])

            df = df.set_index("local_15min")

            df["dayoftheweek"] = df.index.dayofweek
            df["hour"] = df.index.hour
            df["minute"] = df.index.minute

            # df = df.reset_index(drop=True)

            dfs.append(df)

            agg_df_avg = df.groupby(['dayoftheweek', 'hour', 'minute']).agg({"use": "mean"})

            temp_df = agg_df_avg['use']

            plot = temp_df.plot(title=filename + " use aggregated")
            plot.plot()
            plt.show()

    full_agg_df = pd.concat(dfs).groupby("local_15min").mean()

    agg_df_avg = full_agg_df.groupby(['dayoftheweek', 'hour', 'minute']).agg({"use": "mean"})
    agg_df_sum = full_agg_df.groupby(['dayoftheweek', 'hour', 'minute']).agg({"use": "sum"})

    temp_df = agg_df_avg['use']

    plot = temp_df.plot(title="full avg use aggregated")
    plot.plot()
    plt.show()

    temp_df = agg_df_sum['use']

    plot = temp_df.plot(title="full sum use aggregated")
    plot.plot()
    plt.show()


agg_data_and_visualize()
