import os

import pandas as pd
from datetime import datetime

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

time_formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S+%f"]


def print_difference_between_missing_data(file, tdelta, t_colname, tformat=None):
    """
    Function to print differences between missing data points
    :param file: path of the file to parse, must be .csv
    :param tdelta: difference between data points, in seconds
    :param t_colname: name of column in .csv file containing timestamps
    :param tformat: format to parse the timestamp from t_colname
    """
    #  See https://stackabuse.com/how-to-format-dates-in-python/ for good explanation on datetime formatting
    dataset = pd.read_csv(file)
    if not tformat:
        tformat = find_format(dataset.iloc[0][t_colname])
    for index, row in dataset.iterrows():
        if index == 0:
            prev_time = row[t_colname]
            continue
        diff = datetime.strptime(row[t_colname], tformat) - datetime.strptime(prev_time, tformat)
        if diff.total_seconds() == tdelta:
            prev_time = row[t_colname]
            continue
        else:
            print(prev_time, "vs", row[t_colname])
            print(diff)
            prev_time = row[t_colname]


def calculate_percentage_missing_data(file, tdelta, t_colname, tformat=None):
    """
    Function to calculate percentage data missing
    :param file: path of the file to parse, must be .csv
    :param tdelta: difference between data points, in seconds
    :param t_colname: name of column in .csv file containing timestamps
    :param tformat: format to parse the timestamp from t_colname
    """
    #  See https://stackabuse.com/how-to-format-dates-in-python/ for good explanation on datetime formatting
    correct = 0
    incorrect = 0
    dataset = pd.read_csv(file)
    if not tformat:
        tformat = find_format(dataset.iloc[0][t_colname])
    for index, row in dataset.iterrows():
        if index == 0:
            prev_time = row[t_colname]
            continue
        diff = datetime.strptime(row[t_colname], tformat) - datetime.strptime(prev_time, tformat)
        if diff.total_seconds() == tdelta:
            correct += 1
            prev_time = row[t_colname]
            continue
        else:
            incorrect += diff.total_seconds() / tdelta
            prev_time = row[t_colname]
    return incorrect / (correct + incorrect)


def find_format(date_text):
    """
    Helper function to find the datetime format to parse timestamps
    :param date_text: timestamp to parse
    :return: A suitable format
    :raises: Exception if no format can be found
    """
    for time_format in time_formats:
        try:
            datetime.strptime(date_text, time_format)
            return time_format
        except ValueError:
            continue
    raise Exception("No suitable format could be found for", date_text)


def analyse_building_energy_data(folder="data/raw/building_energy/"):
    """
    Small function to analyse data per building energy file, made to be easily expanded if desired.
    """
    print("========= BUILDING ENERGY USAGE ANALYSIS IN " + folder + " =========")
    for filename in os.listdir(folder):
        if ".csv" in filename:
            print("========= Analysis " + filename + " =========")
            # print_difference_between_missing_data(os.path.join(folder, filename), 900, "local_15min")
            print("Percentage missing data:", str(calculate_percentage_missing_data(os.path.join(folder, filename), 900, "local_15min") * 100) + "%")


def clean_data_on_time_range(file, t_colname, start, end, freq, output_folder):
    """
    Fill in missing rows by datetime in pandas and fill NaNs with 0.
    :param file: path of file to insert missing data into, must be .csv
    :param start: Start date, can be formatted like: '1/1/2014'
    :param end: End date, can be formatted like: '31/12/2014'
    :param freq: Frequency between timestamps, for aliases that can be used see here: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
    :param output_folder: Folder to output the time cleaned file to.
    """
    # Make custom time range used to fill missing data
    idx = pd.date_range(start=start, end=end, freq=freq)

    # Read dataset to clean and setup the index
    dataset = pd.read_csv(file)
    dataset = dataset.set_index(t_colname)
    dataset.index = pd.DatetimeIndex(dataset.index)

    # Drop any possible duplicates
    dataset = dataset.drop_duplicates()

    # Reindex dataset, filling missing data with NaNs from custom time range
    dataset = dataset.reindex(idx, fill_value=None)

    # Interpolate the missing data
    dataset = dataset.interpolate()
    dataset.index.name = t_colname

    # Fill any last NaN with 0s
    dataset = dataset.fillna(0)

    # Write the file
    dataset.to_csv(os.path.join(output_folder, "tc-" + file.split("/")[-1]))


def time_clean_building_energy(input_folder="data/raw/building_energy/", output_folder="data/cleaned/building_energy/"):
    """
    Go through files in raw building energy folder and time clean them
    :param input_folder:
    """
    print("========= Performing time cleaning on " + input_folder + " =========")
    for filename in os.listdir(input_folder):
        if ".csv" in filename:
            print("Time cleaning", filename)
            clean_data_on_time_range(file=os.path.join(input_folder, filename), t_colname="local_15min", start='1/1/2014', end='31/12/2015', freq="15T", output_folder=output_folder)


# time_clean_building_energy()
#
#
#
# analyse_building_energy_data()
# analyse_building_energy_data("data/cleaned/building_energy/")

# print_difference_between_missing_data("data/weather1415.csv", 3600, "localhour")
# print(calculate_percentage_missing_data("data/weather1415.csv", 3600, "localhour"))

# df = pd.read_csv("data/cleaned/building_energy/tc-114-building_data-2014.csv")
# print(df.head(5))





