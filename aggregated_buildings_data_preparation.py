import os

import numpy as np
import pandas as pd
import pickle
import data_cleaning
from datetime import datetime

# Set some display options to make viewing df.head() show more.
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500000)
pd.set_option('display.width', 10000000)

# Define amount in time for creating proper features for time
hours_in_day = 24
days_in_week = 7
months_in_year = 12

# Aggregation_kernel_size
aggregation_kernel_size = 121

# When changing this also change in data_cleaning.py
column_data_to_predict, column_data_to_predict_name = [0], 'use'  # 0 is use column, 28 is grid column

weather_columns = ['localhour', 'temperature', 'dew_point', 'humidity', 'visibility', 'apparent_temperature', 'pressure', 'wind_speed', 'cloud_cover', 'precip_intensity', 'precip_probability']


def clean_and_prep_metadata(meta_df, building_id):
    """
    Prepare the metadata
    :param meta_df: metadata dataframe
    :param building_id: current building id
    :return: cleaned and prepared metadata
    """
    # Drop all other buildings
    meta_df = meta_df[meta_df.dataid == building_id]

    # Change NaN and yes to 0 and 1s
    meta_df = meta_df.fillna(0)
    meta_df = meta_df.replace(to_replace="yes", value=1)

    return meta_df


def clean_and_prepare_weather_data(weather_df):
    """
    Clean and prepare the weather data. Remove duplicates and fill missing spots. Then write the prepared data. And add
    data for 15 min intervals.
    :param weather_df: The weather dataframe
    """
    # Get start and endpoint of the weather
    start = datetime.strptime(weather_df.iloc[0]["localhour"], "%Y-%m-%d %H:%M:%S+%f")
    end = datetime.strptime(weather_df.iloc[-1]["localhour"], "%Y-%m-%d %H:%M:%S+%f")

    # Create custom date range in 15 min intervals, this will be merged with current data eventually to introduce the
    # non-existing 15 min intervals.
    idx = pd.date_range(start=start, end=end, freq='15T')

    # Filter columns so we only have the ones we want
    weather_df = weather_df[weather_columns]

    # Drop any duplicates
    weather_df = weather_df.drop_duplicates(subset="localhour")

    # Hack around to make sure the date is in the correct format
    weather_df['localhour'] = pd.to_datetime(weather_df['localhour'])
    weather_df['localhour'] = weather_df['localhour'].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Set index and drop possible duplicates
    weather_df = weather_df.set_index('localhour')
    weather_df.index = pd.DatetimeIndex(weather_df.index)

    # Reindex our weather data with the 15 min interval data
    weather_df = weather_df.reindex(idx, fill_value=None)

    # Interpolate the introduced 15 min intervals
    weather_df = weather_df.interpolate()

    # Change name, as the intervals are 15 min now
    weather_df.index.name = 'local_15min'

    # Fill all NaNs with 0 (if they are left)
    weather_df = weather_df.fillna(0)

    # Write the file
    weather_df.to_csv("data/cleaned/weather/tc-weather-{}-{}.csv".format(start, end))

    return weather_df


def merge_energy_data_with_weather_data(energy_df, path_to_weather_data=None, path_to_tc_weather_data=None):
    """
    Merge the energy data with the weather data. There is energy data every 15 min, and weather data every hour. The
    weather data is first cleaned and 15 min intervals are added, interpolating the data.
    :param energy_df: Energy dataframe
    :param path_to_weather_data: Path to weather csv
    :return: The merged Dataframe
    """
    if path_to_tc_weather_data:
        weather_df = pd.read_csv(path_to_tc_weather_data)
        weather_df = weather_df.set_index('local_15min')
    else:
        if path_to_weather_data:
            # First load and clean weather data
            weather_df = pd.read_csv(path_to_weather_data)
            weather_df = clean_and_prepare_weather_data(weather_df)
        else:
            raise Exception("Path to (cleaned) weather data must be given!")

    # # Get start and end of energy data
    # try:
    #     tformat = data_cleaning.find_format(str(energy_df.iloc[0]['local_15min']))
    # except KeyError:
    #     # Ugly hack for if above does not work
    #     energy_df['temp'] = energy_df.index
    #     tformat = data_cleaning.find_format(str(energy_df.iloc[0]['temp']))
    #     energy_df.drop(columns=['temp'])

    if 'local_15min' in energy_df:
        start_section = energy_df.iloc[0]['local_15min']
        end_section = energy_df.iloc[-1]['local_15min']
    else:
        start_section = energy_df.index[0]
        end_section = energy_df.index[-1]

    # Drop all weather rows not within the range of our energy data
    weather_df = weather_df[weather_df.index.isin(pd.date_range(start=start_section, end=end_section, freq='15T'))]

    # If the key 'local_15min' exists in energy_df, make sure it's the correct type and index
    if 'local_15min' in energy_df:
        energy_df['local_15min'] = pd.to_datetime(energy_df['local_15min'])
        energy_df = energy_df.set_index('local_15min')
    else:
        energy_df.index = pd.to_datetime(energy_df.index)

    # Merge the two dataframe, followed by all my other failed attempts
    merged_df = pd.concat([energy_df, weather_df], axis=1)

    # # Change datetime to month, day of the week and hour of the day
    # merged_df['month'] = merged_df.index.month
    # merged_df['weekday'] = merged_df.index.weekday # TODO: In paper mention time is normalized like this https://medium.com/ai%C2%B3-theory-practice-business/top-6-errors-novice-machine-learning-engineers-make-e82273d394db http://blog.davidkaleko.com/feature-engineering-cyclical-features.html
    # merged_df['hour'] = merged_df.index.hour  # TODO: https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/ make it cyclic and https://www.reddit.com/r/MachineLearning/comments/1utxnk/how_do_you_represent_timeofday_in_artificial/

    # Encode month, week and hour with sin and cos to make the data cyclical
    merged_df['month_sin'] = np.sin(2 * np.pi * merged_df.index.month / months_in_year)
    merged_df['month_cos'] = np.cos(2 * np.pi * merged_df.index.month / months_in_year)

    merged_df['weekday_sin'] = np.sin(2 * np.pi * merged_df.index.weekday / days_in_week)
    merged_df['weekday_cos'] = np.cos(2 * np.pi * merged_df.index.weekday / days_in_week)

    merged_df['hour_sin'] = np.sin(2 * np.pi * merged_df.index.hour / hours_in_day)
    merged_df['hour_cos'] = np.cos(2 * np.pi * merged_df.index.hour / hours_in_day)

    # Drop local_15min as we don't need it anymore
    merged_df = merged_df.reset_index()
    merged_df = merged_df.drop(columns=['local_15min'])

    # weather_df['local_15min'] = weather_df['local_15min'].dt.strftime("%Y-%m-%d %H:%M:%S")
    # merged_df = pd.concat([energy_df, weather_df], join='inner', axis=1)
    # merged_df = energy_df.join(weather_df.set_index('local_15min'), 'local_15min', 'outer')
    # merged_df = pd.concat([d.set_index('local_15min') for d in [energy_df, weather_df]], axis=1).reset_index()
    # merged_df = energy_df.set_index('local_15min').combine_first(weather_df.set_index('local_15min')).reset_index()
    # merged_df = pd.merge(energy_df, weather_df, on="local_15min")

    return merged_df


def prepare_data(path_to_energy_data_folder, path_to_weather_data, output_folder, cleaned_dfs=None):
    """
    Function to prepare all the data, assuming it has already been cleaned.
    :param path_to_energy_data_folder: Path to folder containing energy data over time
    :param path_to_metadata: Path to metadata of the buildings
    :param path_to_weather_data: Path to hourly weather data
    :param output_folder: Folder to output prepared data
    :param cleaned_dfs: Optional parameter to already give the cleaned dataframes instead of having to read them out.
    """
    print("========= Preparing aggregated data as input for NN to " + output_folder + " =========")

    if not cleaned_dfs:
        cleaned_dfs = []
        for filename in os.listdir(path_to_energy_data_folder):
            if ".csv" in filename:
                cleaned_dfs.append(pd.read_csv(os.path.join(path_to_energy_data_folder, filename)))

    # Drop the dataid column
    for cleaned_df in cleaned_dfs:
        cleaned_df.drop(columns=['dataid'], inplace=True)

    total = len(cleaned_dfs) - aggregation_kernel_size

    for i in range(total + 1):
        print("Preparing cleaned aggregated dataset {} of {}".format(i, total))

        # This can easily be optimized but I choose to do it this way to change things more easily
        energy_dfs = []
        for j in range(aggregation_kernel_size):
            energy_dfs.append(cleaned_dfs[i + j])

        aggregated_df = energy_dfs.pop()
        for energy_df in energy_dfs:
            # Set index if it isn't yet
            if 'local_15min' in aggregated_df:
                aggregated_df = aggregated_df.set_index('local_15min')
            if 'local_15min' in energy_df:
                energy_df = energy_df.set_index('local_15min')
            aggregated_df = aggregated_df.add(energy_df)

        prepared_df = merge_energy_data_with_weather_data(aggregated_df, path_to_weather_data)

        prepared_df.to_csv(os.path.join(output_folder, "p-agg-" + str(i) + ".csv"), index=False)


def normalize_data(path_to_data):
    """
    Normalize and collect data into an array of matrices (np), also add the expect output data alongside it
    :param path_to_data: Path to the data
    :return: Collected and normalized data and factors to denormalize output
    """
    print("========= Processing data as input for NN in " + path_to_data + " =========")
    collected_data = []

    # Loop over prepared data, collecting it into numpy arrays.
    for filename in os.listdir(path_to_data):
        if ".csv" in filename:
            print("Processing", filename)
            data = np.genfromtxt(os.path.join(path_to_data, filename), delimiter=',', dtype=np.float32)

            # Delete the first row (names of the columns)
            data = np.delete(data, 0, axis=0)

            # If for some reason any data is still NaN, set it to 0.
            if np.isnan(data).any():
                print("WARNING:", filename, "contains NaN values in " + str(len(np.argwhere(np.isnan(data)))) + " cells")
                data[np.isnan(data)] = 0
            collected_data.append(data)
    print("============================")

    prev_shape = np.shape(collected_data[0])[0]
    matching_shapes = True

    for building in collected_data:
        current_shape = np.shape(building)[0]
        if not prev_shape == current_shape:
            matching_shapes = False
            break
        prev_shape = current_shape

    # Calculate mean and std of each column so we can normalize the data.
    if matching_shapes:
        stacked_collected_data = np.stack(collected_data)
        collected_data_mean = np.mean(stacked_collected_data, axis=(0, 1))
        collected_data_std = np.std(stacked_collected_data, axis=(0, 1))
    else:
        concatenated_collected_data = np.concatenate(collected_data, axis=0)
        collected_data_mean = np.mean(concatenated_collected_data, axis=0)
        collected_data_std = np.std(concatenated_collected_data, axis=0)

    # Loop over calculated mean and std. Change the std to 1 if the column in the data was fully static, since if the
    # column is static it must be either 1 or 0. And dividing by 0 is of course not allowed.
    for i in range(len(collected_data_std)):
        if collected_data_std[i] == 0:
            collected_data_std[i] = 1
            collected_data_mean[i] = 0

    # If all shapes were matching do numpy magic to return normalized data and data to predict
    if matching_shapes:
        normalized_input_data = (collected_data - collected_data_mean) / collected_data_std
        normalized_output_data = np.take(normalized_input_data, indices=column_data_to_predict, axis=2)
        return normalized_input_data, normalized_output_data, collected_data_std[column_data_to_predict], collected_data_mean[column_data_to_predict]

    normalized_collected_data = []
    # Calculate the normalized data
    for building in collected_data:
        building = building - collected_data_mean / collected_data_std
        normalized_collected_data.append(building)

    data_to_predict = []
    # Grab the column containing the to predict value
    for building in normalized_collected_data:
        to_predict = np.take(building, indices=column_data_to_predict, axis=1)
        data_to_predict.append(to_predict)

    return normalized_collected_data, data_to_predict, collected_data_std[column_data_to_predict], collected_data_mean[column_data_to_predict]


def normalize_and_pickle_prepared_data(prepared_data_folder="data/prepared/aggregated_1415/", pickle_output_path=None):
    """
    Normalize data in the prepared data folder and write it to a pickle
    :param prepared_data_folder: Folder with prepared data
    :param pickle_output_path: Path where to output pickle file
    :return:
    """
    normalized_collected_input_data, normalized_output_data, output_std, output_mean = normalize_data(prepared_data_folder)
    if pickle_output_path:
        pickle_file = open(os.path.join(pickle_output_path, "aggregated_input_data-f{}-ak{}-b{}.pkl".format(
            normalized_collected_input_data.shape[2],
            aggregation_kernel_size,
            normalized_collected_input_data.shape[0] + aggregation_kernel_size - 1
        )), "wb")
    else:
        print(normalized_collected_input_data.shape)
        pickle_file = open(os.path.join(prepared_data_folder, "aggregated_input_data-f{}-ak{}-b{}.pkl".format(
            normalized_collected_input_data.shape[2],
            aggregation_kernel_size,
            normalized_collected_input_data.shape[0] + aggregation_kernel_size - 1
        )), "wb")
    pickle.dump(({"normalized_input_data": normalized_collected_input_data,
                  "normalized_output_data": normalized_output_data,
                  "output_std": output_std,
                  "output_mean": output_mean}), pickle_file)
    pickle_file.close()


def the_whole_shibang():
    # cleaned_dfs = data_cleaning.time_clean_building_energy(input_folder="data/raw/building_energy/1415/",
    #                                                        output_folder="data/cleaned/building_energy/1415/",
    #                                                        start_section='1/1/2014',
    #                                                        end_section='31/12/2015')
    prepare_data("data/cleaned/building_energy/1415/",
                 "data/weather1415.csv", "data/prepared/aggregated_1415/")
    normalize_and_pickle_prepared_data()


# data_cleaning.time_clean_building_energy()
# prepare_data("data/cleaned/building_energy/", "data/cleaned/metadata/tc-buildings_metadata.csv", "data/weather1415.csv", "data/prepared")
# normalize_data("data/prepared")
# normalize_and_pickle_prepared_data()
the_whole_shibang()

# cleaned_dfs = data_cleaning.time_clean_building_energy()
# prepare_data("data/cleaned/building_energy/", "data/cleaned/metadata/tc-buildings_metadata.csv", "data/weather_all.csv", "data/prepared/")