import pandas as pd
import os 
import random

# load csv
df_path = os.path.join("training_solutions_rev1.csv")
df = pd.read_csv(df_path)

df_reduce = df.sample(n=3000, axis='rows')

