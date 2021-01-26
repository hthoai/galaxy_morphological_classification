import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import random

# load csv file
df_path = os.path.join("training_solutions_rev1.csv")
df = pd.read_csv(df_path, index_col=0)

dom_classes = df.idxmax(axis=1)
df['Dominant Classes'] = dom_classes
plt.hist(dom_classes)
plt.show()

print(dom_classes.unique())

class6_2 = df.index[df['Dominant Classes'] == 'Class6.2'].tolist()
class1_1 = df.index[df['Dominant Classes'] == 'Class1.1'].tolist()
class6_1 = df.index[df['Dominant Classes'] == 'Class6.1'].tolist()
class1_2 = df.index[df['Dominant Classes'] == 'Class1.2'].tolist()
class1_3 = df.index[df['Dominant Classes'] == 'Class1.3'].tolist()

ratio = 0.5

class6_2_redu = random.sample(class6_2, k=int(len(class6_2)*ratio))
class1_1_redu = random.sample(class1_1, k=int(len(class1_1)*ratio))
class6_1_redu = random.sample(class6_1, k=int(len(class6_1)*ratio))
class1_2_redu = random.sample(class1_2, k=int(len(class1_2)*ratio))
class1_3_redu = random.sample(class1_3, k=int(len(class1_3)*ratio))