import numpy as np
from keras.losses import mean_absolute_percentage_error

from metrics import mean_error
from utils import load_data

timeseries = 10
batch_size = 256

# Define the amount of features in the input and the output
input_feature_amount = 83  # 87 without static indicators, 154 with. If aggregated, 83
output_feature_amount = 1

# Define size of states used by GRU
state_size = 96

# Input and output length sequence (24 * 4 = 96 15 minute intervals in 24 hours)
seq_len_in = 96 * 7
seq_len_out = 96

# TODO: Try out regularization

plot_last_time_steps_view = 96 * 2

test_train_ratio = 0.5

data_dict = load_data("/home/mauk/Workspace/energy_prediction/data/prepared/input_data-f83-0206.pkl")  # "/home/mauk/Workspace/energy_prediction/data/prepared/input_data-f83-3105.pkl")
normalized_input_data = data_dict['normalized_input_data']
normalized_output_data = data_dict['normalized_output_data']
output_std = data_dict['output_std']
output_mean = data_dict['output_mean']


validation_metrics = [mean_error
           # mean_absolute_percentage_error
           # ks.losses.mean_absolute_error
                      ]

# print(normalized_input_data[0][0])


def generate_validation_data(slice_point=1500):
    """
    Generate validation data of the whole testing set.
    :return: the validation data
    """
    test_xe_batches = []
    test_xd_batches = []
    test_y_batches = []

    # Split into testing set
    if hasattr(normalized_input_data, 'shape') and hasattr(normalized_output_data, 'shape'):
        test_x, test_y = normalized_input_data[:, -normalized_input_data.shape[1] // int((1 / test_train_ratio)):], \
                         normalized_output_data[:, -normalized_output_data.shape[1] // int((1 / test_train_ratio)):]
    else:
        test_x = []
        test_y = []
        for building in normalized_input_data:
            test_x.append(building[-int(np.shape(building)[0] * test_train_ratio):])  # Ratio, TODO: Unhardcode, add variable up top
        for building in normalized_output_data:
            test_y.append(building[-int(np.shape(building)[0] * test_train_ratio):])

    for i in range(len(test_x)):
        for j in range(len(test_x[i]) - seq_len_out - seq_len_in):
            #  Change modulo operation to change interval
            if j % slice_point == 0:
                test_xe_batches.append(test_x[i][j:j+seq_len_in])
                test_xd_batches.append(test_y[i][j+seq_len_in - 1:j+seq_len_in+seq_len_out - 1])
                test_y_batches.append(test_y[i][j + seq_len_in:j + seq_len_in + seq_len_out])

    test_xe_batches = np.stack(test_xe_batches, axis=0)
    test_xd_batches = np.stack(test_xd_batches, axis=0)
    test_y_batches = np.stack(test_y_batches, axis=0)

    return [test_xe_batches, test_xd_batches], test_y_batches


def generate_batches():
    """
    Generate batch to be used in training
    :return: Batch for encoder and decoder inputs and a batch for output
    """
    # Split into training set
    if hasattr(normalized_input_data, 'shape') and hasattr(normalized_output_data, 'shape'):
        train_x, train_y = normalized_input_data[:, :normalized_input_data.shape[1]//int((1 / test_train_ratio))], normalized_output_data[:, :normalized_output_data.shape[1] // int((1 / test_train_ratio))]
    else:
        train_x = []
        train_y = []
        for building in normalized_input_data:
            train_x.append(building[:int(len(building) * (1 - test_train_ratio))])  # Ratio, TODO: Unhardcode, add variable up top
        for building in normalized_output_data:
            train_y.append(building[:int(len(building) * (1 - test_train_ratio))])

    while True:
        # Batch input for encoder
        batch_xe = []
        # Batch input for decoder for guided training
        batch_xd = []
        # Batch output
        batch_y = []

        for i in range(batch_size):
            # Select a random building from the training set
            bd = np.random.randint(0, timeseries)

            # Grab a random starting point from 0 to length of dataset - input length encoder - input length decoder
            sp = np.random.randint(0, len(train_x[bd]) - seq_len_in - seq_len_out)

            # Append samples to batches
            batch_xe.append(train_x[bd][sp:sp+seq_len_in])
            batch_xd.append(train_y[bd][sp+seq_len_in-1:sp+seq_len_in+seq_len_out-1])
            batch_y.append(train_y[bd][sp+seq_len_in:sp+seq_len_in+seq_len_out])

        # Stack batches and yield them
        batch_xe = np.stack(batch_xe)
        batch_xd = np.stack(batch_xd)
        batch_y = np.stack(batch_y)
        yield [batch_xe, batch_xd], batch_y


def generate_batch():
    """
    Generate single batch
    :return: Batch for encoder and decoder inputs and a batch for output
    """
    # Split into training set
    if hasattr(normalized_input_data, 'shape') and hasattr(normalized_output_data, 'shape'):
        train_x, train_y = normalized_input_data[:, :normalized_input_data.shape[1]//int((1 / test_train_ratio))], normalized_output_data[:, :normalized_output_data.shape[1] // int((1 / test_train_ratio))]
    else:
        train_x = []
        train_y = []
        for building in normalized_input_data:
            train_x.append(building[:int(len(building) * (1 - test_train_ratio))])  # Ratio, TODO: Unhardcode, add variable up top
        for building in normalized_output_data:
            train_y.append(building[:int(len(building) * (1 - test_train_ratio))])

    # Batch input for encoder
    batch_xe = []
    # Batch input for decoder for guided training
    batch_xd = []
    # Batch output
    batch_y = []

    for i in range(batch_size):
        # Select a random building from the training set
        bd = np.random.randint(0, timeseries)

        # Grab a random starting point from 0 to length of dataset - input length encoder - input length decoder
        sp = np.random.randint(0, len(train_x[bd]) - seq_len_in - seq_len_out)

        # Append samples to batches
        batch_xe.append(train_x[bd][sp:sp+seq_len_in])
        batch_xd.append(train_y[bd][sp+seq_len_in-1:sp+seq_len_in+seq_len_out-1])
        batch_y.append(train_y[bd][sp+seq_len_in:sp+seq_len_in+seq_len_out])

    # Stack batches and yield them
    batch_xe = np.stack(batch_xe)
    batch_xd = np.stack(batch_xd)
    batch_y = np.stack(batch_y)
    return [batch_xe, batch_xd], batch_y


def generate_validation_sample():
    """
    Generate batch to be used for validation, also return the previous ys so we can plot the input as well
    :return: Batch for encoder and decoder inputs and a batch for output
    """
    # Split into testing set
    if hasattr(normalized_input_data, 'shape') and hasattr(normalized_output_data, 'shape'):
        #TODO: Check if this needs minus
        test_x, test_y = normalized_input_data[:, -normalized_input_data.shape[1]//int((1 / test_train_ratio)):], normalized_output_data[:, -normalized_output_data.shape[1] // int((1 / test_train_ratio)):]
    else:
        test_x = []
        test_y = []
        for building in normalized_input_data:
            test_x.append(building[-int(len(building) * test_train_ratio):])  # Ratio, TODO: Unhardcode, add variable up top
        for building in normalized_output_data:
            test_y.append(building[-int(len(building) * test_train_ratio):])

    # Batch input for encoder
    batch_xe = []
    # Batch input for decoder for guided training
    batch_xd = []
    # Batch output
    batch_y = []

    # Select a random building from the training set
    bd = np.random.randint(0, len(test_x))

    # Grab a random starting point from 0 to length of dataset - input length encoder - input length decoder
    sp = np.random.randint(0, len(test_x[bd]) - seq_len_in - seq_len_out)

    # Append sample to batch
    batch_xe.append(test_x[bd][sp:sp + seq_len_in])
    batch_xd.append(test_y[bd][sp + seq_len_in - 1:sp + seq_len_in + seq_len_out - 1])
    batch_y.append(test_y[bd][sp + seq_len_in:sp + seq_len_in + seq_len_out])

    # Output during input frames
    batch_y_prev = test_y[bd][sp:sp + seq_len_in]

    # Stack batches and return them
    batch_xe = np.stack(batch_xe)
    batch_xd = np.stack(batch_xd)
    batch_y = np.stack(batch_y)
    return [batch_xe, batch_xd], batch_y, batch_y_prev


def generate_testing_sample():
    """
    Generate batch to be used for validation, also return the previous ys so we can plot the input as well
    :return: Batch for encoder and decoder inputs and a batch for output
    """
    # Split into training set
    if hasattr(normalized_input_data, 'shape') and hasattr(normalized_output_data, 'shape'):
        test_x, test_y = normalized_input_data[:, :normalized_input_data.shape[1]//int((1 / test_train_ratio))], normalized_output_data[:, :normalized_output_data.shape[1] // int((1 / test_train_ratio))]
    else:
        test_x = []
        test_y = []
        for building in normalized_input_data:
            test_x.append(building[-int(len(building) * test_train_ratio):])  # Ratio, TODO: Unhardcode, add variable up top
        for building in normalized_output_data:
            test_y.append(building[-int(len(building) * test_train_ratio):])

    # Batch input for encoder
    batch_xe = []
    # Batch input for decoder for guided training
    batch_xd = []
    # Batch output
    batch_y = []

    # Select a random building from the training set
    bd = np.random.randint(0, len(test_x))

    # Grab a random starting point from 0 to length of dataset - input length encoder - input length decoder
    sp = np.random.randint(0, len(test_x[bd]) - seq_len_in - seq_len_out)

    # Append sample to batch
    batch_xe.append(test_x[bd][sp:sp + seq_len_in])
    batch_xd.append(test_y[bd][sp + seq_len_in - 1:sp + seq_len_in + seq_len_out - 1])
    batch_y.append(test_y[bd][sp + seq_len_in:sp + seq_len_in + seq_len_out])

    # Output during input frames
    batch_y_prev = test_y[bd][sp:sp + seq_len_in]

    # Stack batches and return them
    batch_xe = np.stack(batch_xe)
    batch_xd = np.stack(batch_xd)
    batch_y = np.stack(batch_y)

    return [batch_xe, batch_xd], batch_y, batch_y_prev

#TODO: WHEN TESTING CURRENT TRAINING BATCH RE-PREPARE THE DATA TO NOT INCLUDE HOUR OF THE DAY!!!!
