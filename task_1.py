import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Superstore_sales_dataset.csv', encoding='latin1')
print(df.head())
print(df.info())