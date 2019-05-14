# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
import pickle
import numpy as np

input_data = open("../data/prepared/input_data.pkl", "rb")
normalized_input_data, output_data = pickle.load(input_data)
print(normalized_input_data, output_data)
print(np.shape(normalized_input_data), np.shape(output_data))

train_x, train_y = normalized_input_data[:, :normalized_input_data.shape[1]//2], output_data[:, :output_data.shape[1]//2]
test_x, test_y = normalized_input_data[:, normalized_input_data.shape[1]//2:], output_data[:, output_data.shape[1]//2:]

print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)