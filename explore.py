import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from wrangle_telco import get_telco_data
from wrangle_telco import prep_telco_data
from wrangle_telco import clean_telco_data
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


# plot variables Function
def plot_variable_pairs(df, columns, hue=None):
    sns.pairplot(df[columns], hue=hue, corner=True, kind= 'reg', plot_kws = {'line_kws':{'color':'red'},'scatter_kws':{'s': 1, 'alpha': 0.5}})
    plt.show()

# Month to Years Function
def months_to_years(df):
    df['tenure_years'] = round(df.tenure // 12)
    df['tenure_years'] = df.tenure_years.astype('int')
    return df

# Plot categorical and continuous variables
def plot_categorical_and_continuous_vars(df, cat_cols, con_cols):
    for con in con_cols:
        for cat in cat_cols:
            fig = plt.figure(figsize = (14, 4))
            fig.suptitle(f'{con} v. {cat}')

            plt.subplot(1, 3, 1)
            sns.boxplot(data = df, x = cat, y = con)
            plt.axhline(df[con].mean())

            plt.subplot(1, 3, 2)
            sns.lineplot(data = df, x = cat, y = con)
            plt.axhline(df[con].mean())

            plt.subplot(1, 3, 3)
            sns.histplot(data = df, x = con, bins = 10, hue = cat)
            plt.show()
