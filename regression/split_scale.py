import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split


def split_my_data(df, train_size = .80, random_state = 123):
    return train_test_split(df[['monthly_charges','tenure','total_charges']], train_size = .80, random_state = 123)


def standard_scaler(train, test):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled


def scale_inverse(scaler, train_scaled, test_scaled):
    train = pd.DataFrame(scaler.inverse_transform(train_scaled), columns=train_scaled.columns.values).set_index([train_scaled.index.values])
    test = pd.DataFrame(scaler.inverse_transform(test_scaled), columns=test_scaled.columns.values).set_index([test_scaled.index.values])
    return scaler, train, test


def uniform_scaler(train, test, seed=123):
    scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state= 123, copy=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled