import pandas as pd

df = pd.read_csv("filtered_combined_building_data.csv")

print("Splitting combined data")
for building, df_building in df.groupby("dataid"):
    print("Processing", building)
    df_building.to_csv("../" + str(building) + "-building_data-1415.csv", index=False)
