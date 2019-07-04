import json
from textblob import TextBlob
import re
import os
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

project_path = '/home/wenting/PycharmProjects/thesis/'
case_centered_sentiment = pd.read_csv(project_path + 'data/target_data/case_centered_sentiment.csv')

# correlations as a scatter matrix
df = case_centered_sentiment.iloc[:, 1:]
df.corr()
scatter_matrix(df, figsize=(6, 6))
plt.show()

# correlation matrix
plt.matshow(df.corr())
plt.xticks(range(len(df.columns)), df.columns)
plt.yticks(range(len(df.columns)), df.columns)
plt.colorbar()
plt.show()
