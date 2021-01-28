import pandas as pd
import os 
import random

def load_csv(filename, num_sample):
    df_path = os.path.join(filename)
    df = pd.read_csv(df_path)
    return df.sample(n=num_sample, axis='rows')