import numpy as np
from keras.losses import mean_absolute_percentage_error

from metrics import mean_error
from utils import load_data

buildings = 15
batch_size = 512

# Define the amount of features in the input and the output
input_feature_amount = 84  # 84 without static indicators, 151 with.
output_feature_amount = 1

# Define size of states used by GRU
state_size = 128

# Input and output length sequence (24 * 4 = 96 15 minute intervals in 24 hours)
seq_len_in = 96
seq_len_out = 96

plot_last_time_steps_view = 96 * 2

normalized_input_data, output_data = load_data("/home/mauk/Workspace/energy_prediction/data/prepared/input_data-f84-0306.pkl")  # "/home/mauk/Workspace/energy_prediction/data/prepared/input_data-f83-3105.pkl")

validation_metrics = [mean_error
           # mean_absolute_percentage_error
           # ks.losses.mean_absolute_error
           ]

# print(normalized_input_data[0][0])

# TODO: reprepare data for missing building 2818


def generate_validation_data():
    """
    Generate validation data of the whole testing set.
    :return: the validation data
    """
    test_xe_batches = []
    test_xd_batches = []
    test_y_batches = []

    # TODO: Doesn't this take training + testing??? fix pls
    test_x, test_y = normalized_input_data, output_data # normalized_input_data[:, normalized_input_data.shape[1]//4], output_data[:, output_data.shape[1]//4]

    for i in range(len(normalized_input_data)):
        for j in range(len(normalized_input_data[i]) - seq_len_out - seq_len_in):
            #  Change modulo operation to change interval
            if j % 1500 == 0:
                test_xe_batches.append(test_x[i][j:j+seq_len_in])
                test_xd_batches.append(test_y[i][j+seq_len_in - 1:j+seq_len_in+seq_len_out - 1])
                test_y_batches.append(test_y[i][j + seq_len_in:j + seq_len_in + seq_len_out])

    test_xe_batches = np.stack(test_xe_batches, axis=0)
    test_xd_batches = np.stack(test_xd_batches, axis=0)
    test_y_batches = np.stack(test_y_batches, axis=0)

    # print("xe", np.shape(test_xe_batches))
    # print("xd", np.shape(test_xd_batches))
    # print("y", np.shape(test_y_batches))

    return [test_xe_batches, test_xd_batches], test_y_batches


def generate_batches():
    """
    Generate batch to be used in training
    :return: Batch for encoder and decoder inputs and a batch for output
    """
    # Split into training set
    if hasattr(normalized_input_data, 'shape') and hasattr(output_data, 'shape'):
        train_x, train_y = normalized_input_data[:, :normalized_input_data.shape[1]//4], output_data[:, :output_data.shape[1]//4]
    else:
        train_x = []
        train_y = []
        for building in normalized_input_data:
            train_x.append(building[:int(len(building) * 0.75)])  # Ratio, TODO: Unhardcode, add variable up top
        for building in output_data:
            train_y.append(building[:int(len(building) * 0.75)])

    while True:
        # Batch input for encoder
        batch_xe = []
        # Batch input for decoder for guided training
        batch_xd = []
        # Batch output
        batch_y = []

        for i in range(batch_size):
            # Select a random building from the training set
            bd = np.random.randint(0, buildings)

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
    if hasattr(normalized_input_data, 'shape') and hasattr(output_data, 'shape'):
        train_x, train_y = normalized_input_data[:, :normalized_input_data.shape[1]//4], output_data[:, :output_data.shape[1]//4]
    else:
        train_x = []
        train_y = []
        for building in normalized_input_data:
            train_x.append(building[:int(len(building) * 0.75)])  # Ratio, TODO: Unhardcode, add variable up top
        for building in output_data:
            train_y.append(building[:int(len(building) * 0.75)])

    # Batch input for encoder
    batch_xe = []
    # Batch input for decoder for guided training
    batch_xd = []
    # Batch output
    batch_y = []

    for i in range(batch_size):
        # Select a random building from the training set
        bd = np.random.randint(0, buildings)

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
    if hasattr(normalized_input_data, 'shape') and hasattr(output_data, 'shape'):
        #TODO: Check if this needs minus
        test_x, test_y = normalized_input_data[:, normalized_input_data.shape[1]//4:], output_data[:, output_data.shape[1]//4:]
    else:
        test_x = []
        test_y = []
        for building in normalized_input_data:
            test_x.append(building[-int(len(building) * 0.25):])  # Ratio, TODO: Unhardcode, add variable up top
        for building in output_data:
            test_y.append(building[-int(len(building) * 0.25):])

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
    if hasattr(normalized_input_data, 'shape') and hasattr(output_data, 'shape'):
        test_x, test_y = normalized_input_data[:, :normalized_input_data.shape[1]//4], output_data[:, :output_data.shape[1]//4]
    else:
        test_x = []
        test_y = []
        for building in normalized_input_data:
            test_x.append(building[-int(len(building) * 0.25):])  # Ratio, TODO: Unhardcode, add variable up top
        for building in output_data:
            test_y.append(building[-int(len(building) * 0.25):])

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
