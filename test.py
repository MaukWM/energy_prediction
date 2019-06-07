import os

import pandas as pd

# Set some display options to make viewing df.head() show more.
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500000)
pd.set_option('display.width', 10000000)

dir = "/home/mauk/Workspace/energy_prediction/data/raw/building_energy/1415/og_15buildings+some/incorrect"

for filename in os.listdir(dir):
    print("Processing", filename)
    if ".csv" in filename:
        df = pd.read_csv(os.path.join(dir, filename))
        if "Unnamed: 0" in df.columns:
            df.drop("Unnamed: 0", axis=1, inplace=True)
        if pd.to_datetime(df.loc[0]['local_15min']).date() == pd.to_datetime("2014/01/01 00:00:00.00000").date():
            if pd.to_datetime(df.iloc[-1]['local_15min']).date() == pd.to_datetime("2015/12/31 00:00:00.00000").date():
                df.to_csv(os.path.join("/home/mauk/Workspace/energy_prediction/data/raw/building_energy/1415/", filename),
                          index=False)

