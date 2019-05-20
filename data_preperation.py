import os

import numpy as np
import pandas as pd
import pickle
import data_cleaning

# Set some display options to make viewing df.head() more easy.
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500000)
pd.set_option('display.width', 10000000)

DATA_LENGTH = 69986
column_data_to_predict = [28]  # 0 is use column, 28 is grid column


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


def merge_energy_data_with_metadata(path_to_energy_data, path_to_metadata):
    """
    Function to merge the changing energy data with the static metadata
    :param path_to_energy_data: Path to file containing energy data for a single building
    :param path_to_metadata: Path to metadata of the buildings
    :return: Merged dataframe of the two datasets
    """
    # Load energy data
    energy_df = pd.read_csv(path_to_energy_data)
    building_id = energy_df['dataid'][0]

    # Prepare the meta data
    meta_df = pd.read_csv(path_to_metadata)
    meta_df = clean_and_prep_metadata(meta_df, building_id)

    # Merge the two dataframes
    merged_df = pd.merge(energy_df, meta_df, on="dataid")

    # Drop building id as we don't need it anymore
    merged_df = merged_df.drop("dataid", axis=1)

    return merged_df


def clean_and_prepare_weather_data(weather_df):
    """
    Clean and prepare the weather data. Remove duplicates and fill missing spots. Then write the prepared data. And add
    data for 15 min intervals.
    :param weather_df: The weather dataframe
    """
    # Create custom date range in 15 min intervals, this will be merged with current data eventually to introduce the
    # non-existing 15 min intervals.
    idx = pd.date_range(start='1/1/2014', end='31/12/2015', freq='15T')

    # Hack around to make sure the date is in the correct format
    weather_df['localhour'] = pd.to_datetime(weather_df['localhour'])
    weather_df['localhour'] = weather_df['localhour'].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Set index and drop possible duplicates
    weather_df = weather_df.set_index('localhour')
    weather_df = weather_df.drop_duplicates()
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
    weather_df.to_csv("data/cleaned/weather/tc-weather1415.csv")


def merge_energy_data_with_weather_data(energy_df, path_to_weather_data):
    """
    Merge the energy data with the weather data. There is energy data every 15 min, and weather data every hour. The
    weather data is first cleaned and 15 min intervals are added, interpolating the data.
    :param energy_df: Energy dataframe
    :param path_to_weather_data: Path to weather csv
    :return: The merged Dataframe
    """
    # First load and clean weather data
    weather_df = pd.read_csv(path_to_weather_data)
    clean_and_prepare_weather_data(weather_df)
    weather_df = pd.read_csv("data/cleaned/weather/tc-weather1415.csv")

    # print("START", energy_df.iloc[68388, :])
    # Rename column with time so we can merge the two dataframes (warning: ugly code)
    weather_df = weather_df.rename(columns={"localhour": "local_15min"})
    weather_df['local_15min'] = pd.to_datetime(weather_df['local_15min'])
    energy_df['local_15min'] = pd.to_datetime(energy_df['local_15min'])
    weather_df['local_15min'] = weather_df['local_15min'].dt.strftime("%Y-%m-%d %H:%M:%S")
    weather_df['local_15min'] = pd.to_datetime(weather_df['local_15min'])

    # print("AFTER DATE RENAMING AND STUFF", energy_df.iloc[68388, :])

    # Set the indexes so we know what to merge on
    energy_df = energy_df.set_index('local_15min')
    weather_df = weather_df.set_index('local_15min')
    # print("AFTER DATE INDEXING ENERGY", energy_df.iloc[68388, :], energy_df)
    # print("AFTER DATE INDEXING WEATHER", weather_df.iloc[68388, :])

    # Merge the two dataframe, followed by all my other failed attempts
    merged_df = pd.concat([energy_df, weather_df], axis=1)
    # print("AFTER CONCAT", merged_df.iloc[68388, :])

    # print("AFTER CONCAT", energy_df.iloc[68388, :])
    # Change datetime to month and day of the week
    merged_df['month'] = merged_df.index.month
    merged_df['weekday'] = merged_df.index.weekday

    # print("AFTER SETTING MONTH AND WEEKDAY", merged_df.iloc[68388, :])
    # Drop local_15min as we don't need it anymore
    merged_df = merged_df.reset_index()
    merged_df = merged_df.drop(columns=['local_15min'])
    # print("AFTER DROP LOCAL15", merged_df.iloc[68388, :])

    # weather_df['local_15min'] = weather_df['local_15min'].dt.strftime("%Y-%m-%d %H:%M:%S")
    # merged_df = pd.concat([energy_df, weather_df], join='inner', axis=1)
    # merged_df = energy_df.join(weather_df.set_index('local_15min'), 'local_15min', 'outer')
    # merged_df = pd.concat([d.set_index('local_15min') for d in [energy_df, weather_df]], axis=1).reset_index()
    # merged_df = energy_df.set_index('local_15min').combine_first(weather_df.set_index('local_15min')).reset_index()
    # merged_df = pd.merge(energy_df, weather_df, on="local_15min")
    return merged_df


def prepare_data(path_to_energy_data_folder, path_to_metadata, path_to_weather_data, output_folder):
    """
    Function to prepare all the data, assuming it has already been cleaned.
    :param path_to_energy_data_folder: Path to folder containing energy data over time
    :param path_to_metadata: Path to metadata of the buildings
    :param path_to_weather_data: Path to hourly weather data
    :param output_folder: Folder to output prepared data
    """
    print("========= Preparing data as input for NN to " + output_folder + " =========")
    # Prepare each file one by one
    for filename in os.listdir(path_to_energy_data_folder):
        if ".csv" in filename:
            print("Preparing", filename)
            df = merge_energy_data_with_metadata(os.path.join(path_to_energy_data_folder, filename), path_to_metadata)
            # if '1642' in filename:
            #     print(df.iloc[68388, :])
            # indexes = df['use'].index[df['use'].apply(np.isnan)]
            # print("after merging with metadata", filename, [df.index(i) for i in indexes])
            prepared_df = merge_energy_data_with_weather_data(df, path_to_weather_data)
            # indexes = df['use'].index[df['use'].apply(np.isnan)]
            # print("after merging with weather data", filename, [df.index(i) for i in indexes])
            # if '1642' in filename:
            #     print(prepared_df.iloc[68388, :])
            prepared_df.to_csv(os.path.join(output_folder, "p-" + filename), index=False)


def normalize_data(path_to_data):
    """
    Normalize and collect data into an array of matrices (np), also add the expect output data alongside it
    :param path_to_data: Path to the data
    :return: Collected and normalized data
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
                # print(np.argwhere(np.isnan(data)))
                data[np.isnan(data)] = 0
            collected_data.append(data)
    print("============================")

    # Calculate mean and std of each column so we can normalize the data.
    stacked_collected_data = np.stack(collected_data)
    stacked_collected_data_mean = np.mean(stacked_collected_data, axis=(0, 1))
    stacked_collected_data_std = np.std(stacked_collected_data, axis=(0, 1))

    # Loop over calculated mean and std. Change the std to 1 if the column in the data was fully static, since if the
    # column is static it must be either 1 or 0. And dividing by 0 is of course not allowed.
    for i in range(len(stacked_collected_data_std)):
        if stacked_collected_data_std[i] == 0:
            stacked_collected_data_std[i] = 1
            stacked_collected_data_mean[i] = 0
    # print(np.shape(stacked_collected_data))

    # Calculate the normalized data
    return (collected_data - stacked_collected_data_mean) / stacked_collected_data_std, np.take(collected_data, indices=column_data_to_predict, axis=2)


def normalize_and_pickle_prepared_data(prepared_data_folder="data/prepared/", pickle_output_path=None):
    """
    Normalize data in the prepared data folder and write it to a pickle
    :param prepared_data_folder: Folder with prepared data
    :param pickle_output_path: Path where to output pickle file
    :return:
    """
    normalized_collected_input_data, output_data = normalize_data(prepared_data_folder)
    if pickle_output_path:
        pickle_file = open(pickle_output_path, "wb")
    else:
        pickle_file = open(os.path.join(prepared_data_folder, "input_data.pkl"), "wb")
    pickle.dump((normalized_collected_input_data, output_data), pickle_file)
    pickle_file.close()


def the_whole_shibang():
    data_cleaning.time_clean_building_energy()
    prepare_data("data/cleaned/building_energy/", "data/buildings_metadata.csv", "data/weather1415.csv", "data/prepared/")
    normalize_and_pickle_prepared_data()



# prepare_data("data/cleaned/building_energy/", "data/buildings_metadata_filtered.csv", "data/weather1415.csv", "data/prepared")
# normalize_data("data/prepared")
# normalize_and_pickle_prepared_data()
the_whole_shibang()
