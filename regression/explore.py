from pandas_profiling import ProfileReport
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wrangle
import split_scale



def plot_variable_pairs(df):
    sns.pairplot(df, kind="reg")
    plt.show()


def months_to_years(df):
    df['tenure_years']=round(df.tenure/12).astype(int)
    return df


def plot_categorical_and_continuous_vars(df):
    plt.figure(figsize=(16,8))
    plt.subplot(1,3,1)
    plt.bar(df.tenure_years,df.total_charges)
    plt.xlabel('Tenure in years')
    plt.ylabel('Total charges in dollars')
    plt.subplot(1,3,2)
    #sns.catplot(x="tenure_years", y="total_charges", data=df)
    sns.stripplot(df.tenure_years,df.total_charges)
    plt.subplot(1,3,3)
    plt.pie(df.groupby('tenure_years')['total_charges'].sum(),labels=list(df.tenure_years.unique()),autopct='%1.1f%%',shadow=True)
    plt.title(" Percent of total charges by tenure")
    plt.show()

