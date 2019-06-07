from utils import load_data

normalized_input_data, output_data = load_data("/home/mauk/Workspace/energy_prediction/data/prepared/aggregated_input_data-f83-ak5-b15.pkl")  # "/home/mauk/Workspace/energy_prediction/data/prepared/input_data-f83-3105.pkl")

print(normalized_input_data.shape)
print(output_data.shape)

print(normalized_input_data[0][0])

for i in range(len(output_data[0])):
    print(output_data[0][i])
