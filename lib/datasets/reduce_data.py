import pandas as pd
import os 
import random

# load csv
df_path = os.path.join("training_solutions_rev1.csv")
df = pd.read_csv(df_path)
data_id = list(df['GalaxyID'])

# choose k data randomly 
data_reduce = random.sample(data_id, k=3000)
