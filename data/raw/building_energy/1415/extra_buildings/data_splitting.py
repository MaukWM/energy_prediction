import os

import pandas as pd

print("Splitting combined data")
for filename in os.listdir("./"):
    print("Processing", filename)
    if ".csv" in filename:
        df = pd.read_csv(filename)
        for building, df_building in df.groupby("dataid"):
            print("Processing", building)
            df_building.to_csv("../" + str(building) + "-building_data-1415.csv", index=False)
