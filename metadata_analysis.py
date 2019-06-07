import pandas as pd

# Set some display options to make viewing df.head() show more.
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500000)
pd.set_option('display.width', 10000000)

must_have_columns = ['use', 'air1', 'furnace1']

desired_columns = ['air1', 'clotheswasher1', 'dishwasher1', 'kitchenapp1', 'microwave1', 'oven1',
                  'refrigerator1']

df = pd.read_csv("data/dataport-metadata.csv")

# Drop row if use not present
for must_have_column in must_have_columns:
    df = df[df[must_have_column].notnull()]

# Drop row if not in Austin, Texas (for weather data merging)
df = df[(df.city == "Austin") & (df.state == "Texas")]

# Drop row if no data from 2014-01-01 until at least 2015-12-31
df = df[(pd.to_datetime(df.egauge_min_time) < pd.to_datetime('2014-01-01'))
        & (pd.to_datetime(df.egauge_max_time) > pd.to_datetime('2015-12-31'))]

# TODO: Implement threshold
# for desired_column in desired_columns:
#     df = df[df[desired_column].notnull()]


df.to_csv("data/dataport-metadata-filtered.csv")
