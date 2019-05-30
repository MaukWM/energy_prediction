import numpy as np


from utils import load_data

# Define some variables for generating batches
buildings = 15
batch_size = 256

# Input and output length sequence (24 * 4 = 96 15 minute intervals in 24 hours)
seq_len_in = 96 * 2
seq_len_out = 96

normalized_input_data, output_data = load_data()


def generate_validation_data():
    """
    Generate validation data of the whole testing set.
    :return: the validation data
    """
    test_xe_batches = []
    test_xd_batches = []
    test_y_batches = []

    for i in range(len(normalized_input_data)):
        for j in range(len(normalized_input_data[i]) - seq_len_out - seq_len_in):
            #  Change modulo operation to change interval
            if j % 1500 == 0:
                test_xe_batches.append(normalized_input_data[i][j:j+seq_len_in])
                test_xd_batches.append(output_data[i][j+seq_len_in - 1:j+seq_len_in+seq_len_out - 1])
                test_y_batches.append(output_data[i][j + seq_len_in:j + seq_len_in + seq_len_out])

    test_xe_batches = np.stack(test_xe_batches, axis=0)
    test_xd_batches = np.stack(test_xd_batches, axis=0)
    test_y_batches = np.stack(test_y_batches, axis=0)

    return [test_xe_batches, test_xd_batches], test_y_batches


def generate_batches():
    """
    Generate batch to be used in training
    :return: Batch for encoder and decoder inputs and a batch for output
    """
    while True:
        # Split into training and testing set
        train_x, train_y = normalized_input_data[:, :normalized_input_data.shape[1]//2], output_data[:, :output_data.shape[1]//2]

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
    # Split into training and testing set
    train_x, train_y = normalized_input_data[:, :normalized_input_data.shape[1]//2], output_data[:, :output_data.shape[1]//2]

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
    # Split into training and testing set
    test_x, test_y = normalized_input_data[:, normalized_input_data.shape[1]//2:], output_data[:, output_data.shape[1]//2:]

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
    # Split into training and testing set
    test_x, test_y = normalized_input_data[:, :normalized_input_data.shape[1]//2], output_data[:, :output_data.shape[1]//2]

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
