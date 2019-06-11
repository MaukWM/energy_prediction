import os

import pandas as pd
from datetime import datetime

# When changing this also change in data_preparation.py
column_data_to_predict, column_data_to_predict_name = [0], 'use'  # 0 is use column, 28 is grid column

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

time_formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S+%f"]

metadata_columns = ['dataid', 'pv', 'has_electric_vehicle', 'has_quick_charge_port', 'total_square_footage']


def print_difference_between_missing_data(file, tdelta, t_colname, tformat=None):
    """
    Function to print differences between missing data points
    :param file: path of the file to parse, must be .csv
    :param tdelta: difference between data points, in seconds
    :param t_colname: name of column in .csv file containing timestamps
    :param tformat: format to parse the timestamp from t_colname
    """
    #  See https://stackabuse.com/how-to-format-dates-in-python/ for good explanation on datetime formatting
    df = pd.read_csv(file)

    # Remove all rows that contain a NaN value for the predicting column
    df = df[df[column_data_to_predict_name].notnull()]

    if not tformat:
        tformat = find_format(df.iloc[0][t_colname])

    prev_time = df.iloc[0][t_colname]

    for index, row in df.iterrows():
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
    df = pd.read_csv(file)

    # Remove all rows that contain a NaN value for the predicting column
    col_df = df[column_data_to_predict_name]
    print(col_df.isna().sum())
    df = df[(df[column_data_to_predict_name].notnull())]

    if not tformat:
        tformat = find_format(df.iloc[0][t_colname])

    prev_time = df.iloc[0][t_colname]

    for index, row in df.iterrows():
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


def clean_data_on_time_range(t_colname, start, end, freq, output_folder=None, file=None, df=None):
    """
    Fill in missing rows by datetime in pandas and fill NaNs with 0.
    :param file: path of file to insert missing data into, must be .csv
    :param start: Start date, can be formatted like: '1/1/2014'
    :param end: End date, can be formatted like: '31/12/2014'
    :param freq: Frequency between timestamps, for aliases that can be used see here: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
    :param output_folder: Folder to output the time cleaned file to.
    :param write_file: Boolean whether to write the result
    :param df: Optional parameter to use an already existing dataframe
    """
    # Make custom time range used to fill missing data
    idx = pd.date_range(start=start, end=end, freq=freq)

    if df is None:
        if file:
            # Read dataset to clean and setup the index
            df = pd.read_csv(file)
            df = df.set_index(t_colname)
            df.index = pd.DatetimeIndex(df.index)
        else:
            raise Exception("Must give either a dataframe or a file when cleaning on a time range!")

    # Drop any possible duplicates
    df = df.drop_duplicates()

    # Reindex dataset, filling missing data with NaNs from custom time range
    df = df.reindex(idx, fill_value=None)

    # Interpolate the missing data
    df = df.interpolate()
    df.index.name = t_colname

    # Fill any last NaN with 0s
    df = df.fillna(0)

    if output_folder:
        # Write the file
        df.to_csv(os.path.join(output_folder, "tc-" + file.split("/")[-1]))

    return df


def find_largest_section(df, tdelta, t_colname, gap_ratio=16):
    """
    Function to find the largest continuous section (with some wiggle room). Also viewing rows with empty value for
    predicting column as gaps.
    :param df: The dataframe
    :param tdelta: The difference between data points, in seconds
    :param t_colname: The name of column in .csv file containing timestamps
    :param gap_ratio: The ratio threshold
    :return: Tuple with preferred start and end point for df
    """
    # Remove all rows that contain a NaN value for the predicting column
    df = df[df[column_data_to_predict_name].notnull()]

    # Get the proper timeformat for this dataframe
    tformat = find_format(df.iloc[0][t_colname])

    # Define start and end of dataframe
    start = datetime.strptime(df.iloc[0][t_colname], tformat)
    end = datetime.strptime(df.iloc[-1][t_colname], tformat)

    # Define start and end of a section
    start_section = start
    end_section = None

    sections = []

    # Initialize first timestep
    # If we're at index 0, set the previous time
    prev_time = df.iloc[0][t_colname]

    for index, row in df.iterrows():
        diff = datetime.strptime(row[t_colname], tformat) - datetime.strptime(prev_time, tformat)
        if diff.total_seconds() == tdelta:
            prev_time = row[t_colname]
            continue
        else:
            # If the total difference is larger than tdelta * ratio the gap is too large and we have an end/start point
            if diff.total_seconds() > (pd.to_timedelta(tdelta).total_seconds() * gap_ratio):
                if start_section:
                    end_section = datetime.strptime(prev_time, tformat)
                    sections.append(((end_section - start_section).total_seconds(), start_section, end_section))
                    start_section = datetime.strptime(row[t_colname], tformat)
                    end_section = None
                    prev_time = row[t_colname]
                    continue
                else:
                    start_section = datetime.strptime(row[t_colname], tformat)
                    prev_time = row[t_colname]
                    continue
            else:
                prev_time = row[t_colname]
                continue

    # In the end, if there is no end section for the current section, set it to the last entry in our df
    if not end_section:
        end_section = end
        sections.append(((end_section - start_section).total_seconds(), start_section, end_section))

    # Return largest found section
    return max(sections, key=lambda item: item[0])


def clean_data_with_threshold_missing(path_to_file, t_colname, tdelta, output_folder=None):
    """
    Function to clean data on a non-specific time range, but does cut off parts of the data if too much is missing
    :param path_to_file: Path to file
    :param t_colname: Column name of time column
    :param tdelta: The difference between data points, in seconds
    :param output_folder: The folder to output the cleaned data
    :return:
    """
    # Read dataset to clean and setup the index
    df = pd.read_csv(path_to_file)

    # Find the largest section to be used for the data
    _, start_section, end_section = find_largest_section(df, tdelta, t_colname)

    # Set index of dataframe to be the datatime
    df = df.set_index(t_colname)
    df.index = pd.DatetimeIndex(df.index)

    # Drop all rows not within the range of the largest section
    df = df[df.index.isin(pd.date_range(start=start_section, end=end_section, freq=tdelta))]

    df = clean_data_on_time_range(t_colname=t_colname, start=start_section, end=end_section, freq=tdelta, df=df)

    if output_folder:
        # Write the file
        df.to_csv(os.path.join(output_folder, "tc-" + path_to_file.split("/")[-1]))

    return df


def time_clean_building_energy(input_folder="data/raw/building_energy/", output_folder="data/cleaned/building_energy/",
                               start_section=None, end_section=None):
    """
    Go through files in raw building energy folder and time clean them
    :param input_folder:
    :param start_section: Moment in time to start
    :param end_section: Moment in time to end
    """
    print("========= Performing time cleaning on " + input_folder + " =========")
    cleaned_dfs = []
    for filename in os.listdir(input_folder):
        if ".csv" in filename:
            # Clean the dataframe
            print("Time cleaning", filename)
            # clean_data_on_time_range(file=os.path.join(input_folder, filename), t_colname="local_15min", start='1/1/2014', end='31/12/2015', freq="15T", output_folder=output_folder)
            if start_section and end_section:
                cleaned_df = clean_data_on_time_range(file=os.path.join(input_folder, filename),
                                                      t_colname="local_15min", start=start_section, end=end_section,
                                                      freq="15T", output_folder=output_folder)
            else:
                cleaned_df = clean_data_with_threshold_missing(path_to_file=os.path.join(input_folder, filename), t_colname="local_15min",
                                                               tdelta="15T", output_folder=output_folder)

            cleaned_dfs.append(cleaned_df)

    return cleaned_dfs


def clean_building_metadata(path_to_metadata="data/buildings_metadata.csv", output_folder="data/cleaned/metadata/"):
    # Load in the data
    df = pd.read_csv(path_to_metadata)

    # Filter columns so we only have the ones we want
    df = df[metadata_columns]

    # Fill the NaNs
    df = df.fillna(0)

    # Change yes' to 1s
    df = df.replace(to_replace="yes", value=1)

    # Get all dataframes with no square footage indication and drop them
    df_no_sq_ft = df['total_square_footage'] != 0
    df = df[df_no_sq_ft]

    if output_folder:
        # Write the cleaned metadata
        df.to_csv(os.path.join(output_folder, "tc-" + path_to_metadata.split("/")[-1]), index=False)

    return df


# # clean_building_metadata()
# for filename in os.listdir("/home/mauk/Workspace/energy_prediction/data/raw/building_energy/1415/"):
#     if ".csv" in filename:
#         print(filename)
#         # print_difference_between_missing_data(os.path.join("/home/mauk/Workspace/energy_prediction/data/raw/building_energy/1415/", filename), 3600/4, "local_15min")
#         print("%.4f" % (calculate_percentage_missing_data(os.path.join("/home/mauk/Workspace/energy_prediction/data/raw/building_energy/1415/", filename), 3600/4, "local_15min") * 100) + "%")

# time_clean_building_energy()
#
#
#
# analyse_building_energy_data()
# analyse_building_energy_data("data/cleaned/building_energy/")
#

# print_difference_between_missing_data("data/raw/building_energy/4767-building_data.csv", 3600/4, "local_15min")
# print(calculate_percentage_missing_data("data/raw/building_energy/4767-building_data.csv", 3600/4, "local_15min"))

# print_difference_between_missing_data("data/cleaned/weather/tc-weather-2011-01-01 06:00:00-2019-05-24 23:00:00.csv", 3600/4, "local_15min")
# print_difference_between_missing_data("data/weather_all.csv", 3600, "localhour")
# print(calculate_percentage_missing_data("data/weather_all.csv", 3600, "localhour"))
#
# df = pd.read_csv("data/cleaned/building_energy/tc-114-building_data-2014.csv")
# print(df.head(5))





