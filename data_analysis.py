import datetime
import os

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, DateFormatter

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.width', 1000)

weather_columns = ['temperature', 'dew_point', 'humidity', 'visibility', 'apparent_temperature', 'pressure', 'wind_speed', 'cloud_cover', 'precip_intensity', 'precip_probability']


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


# This could be one func, but I'm lazy
def agg_use_data_into_day():
    # Agg all data into one day
    path_to_data_folder = "/home/mauk/Workspace/energy_prediction/data/raw/building_energy/1415/"

    dfs = []
    for filename in os.listdir(path_to_data_folder):
        if ".csv" in filename:
            print("Processing", filename)
            df = pd.read_csv(os.path.join(path_to_data_folder, filename))

            df["local_15min"] = pd.to_datetime(df["local_15min"])

            df = df.set_index("local_15min")

            df["hour"] = df.index.hour
            df["minute"] = df.index.minute

            dfs.append(df)

    full_agg_df = pd.concat(dfs).groupby("local_15min").mean()

    agg_df_avg = full_agg_df.groupby(['hour', 'minute']).agg({"use": "mean"})
    agg_df_sum = full_agg_df.groupby(['hour', 'minute']).agg({"use": "sum"})

    temp_df = agg_df_avg['use']
    temp_df.to_csv("/home/mauk/Workspace/energy_prediction/data/prepared/aggregated_1415/agg121-p/use_vg_agg.csv")

    plot = temp_df.plot(title="full avg use aggregated")
    plot.plot()
    plt.show()

    temp_df = agg_df_sum['use']

    plot = temp_df.plot(title="full sum use aggregated")
    plot.plot()
    plt.show()


def show_agg_avg_data():
    path_to_data = "/home/mauk/Workspace/energy_prediction/data/prepared/aggregated_1415/agg121-p/use_vg_agg.csv"

    df = pd.read_csv(path_to_data)

    customdate = datetime.datetime(2016, 1, 1, 0, 0)
    y = df.iloc[:, 2]
    x = [customdate + datetime.timedelta(minutes=15 * i) for i in range(len(y))]

    ax = plt.subplot()

    # removing top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # adds major gridlines
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    ax.plot(x, y, color='black', linewidth=0.6)
    ax.xaxis.set_major_locator(HourLocator(interval=3))
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.xaxis.set_ticks([customdate + datetime.timedelta(hours=i * 3) for i in range(int(24 / 3) + 1)])
    ax.set_xlabel("Time (15-min interval)")
    ax.set_ylabel("Average energy use [kWh]")

    plt.show()


def calc_mean_and_range_columns(path_to_data, columns):
    df = pd.read_csv(path_to_data)

    result_dict = {}
    for column in columns:
        temp_df = df[column]
        temp_res = dict()
        temp_res['mean'] = temp_df.mean(axis=0)
        temp_res['min'] = df[column].min()
        temp_res['max'] = df[column].max()
        result_dict[column] = temp_res
    return result_dict


res_dict = calc_mean_and_range_columns(path_to_data="/home/mauk/Workspace/energy_prediction/data/weather1415.csv",
                                       columns=weather_columns)

# for item in res_dict:
#     print(item, res_dict[item])

show_agg_avg_data()
# agg_use_data_into_day()
