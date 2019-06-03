from utils import load_data

old_norm_data, old_output_data = load_data(pkl_path="/home/mauk/Workspace/energy_prediction/data/prepared/input_data-f83-3105.pkl")

new_norm_data, new_output_data = load_data(pkl_path="/home/mauk/Workspace/energy_prediction/data/prepared/input_data-f84-0206.pkl")

print("Old data")
for old_norm in old_norm_data:
    print(old_norm.shape)

for old_output in old_output_data:
    print(old_output.shape)

print("New data")
# for new_norm in new_norm_data:
#     print(new_norm.shape)
#
# for new_output in new_output_data:
#     print(new_output.shape)

for i in range(len(new_output_data)):
    print(new_norm_data[i].shape, new_output_data[i].shape)

print(new_norm_data[0][0][0])
print(new_norm_data[0][1][0])
print(new_norm_data[0][2][0])
print(new_norm_data[0][3][0])
print(new_output_data[0][0])
