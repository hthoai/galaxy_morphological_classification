import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 1412
# Read csv
df = pd.read_csv('datasets/training_solutions_rev1.csv')
print(f'df shape: {df.shape}')
# Split
train, test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
train, val = train_test_split(train, test_size=0.2, random_state=RANDOM_STATE)
print(f'train shape: {train.shape}\tval shape: {val.shape}\ttest shape: {test.shape}')
# Save
train.to_csv('datasets/train.csv', index=False)
val.to_csv('datasets/val.csv', index=False)
test.to_csv('datasets/test.csv', index=False)
