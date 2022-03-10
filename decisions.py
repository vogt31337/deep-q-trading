#Copyright (C) 2020 Salvatore Carta, Anselmo Ferreira, Alessandro Sebastian Podda, Diego Reforgiato Recupero, Antonio Sanna. All rights reserved.

import pandas as pd
import numpy as np


def full_ensemble(df):
    """
    Compute the ensemble on columns (nets) with 100% agreement
    :param df:
    :return:
    """
    # Check rows with only 1
    m1 = df.eq(1).all(axis=1)

    # Check rows with only 1
    m2 = df.eq(2).all(axis=1)

    local_df = df.copy()
    # Create a new "ensemble" column which ha 1 if the other rows are all at 1, 0 otherwise.
    local_df['ensemble'] = np.select([m1, m2], [1, -1], 0)

    # remove all colums except the "enseble" one
    local_df = local_df.drop(local_df.columns.difference(['ensemble']), axis=1)

    return local_df


def perc_ensemble(df, thr=0.7):
    """
    Compute ensemble with any % of agreement
    :param df:
    :param thr:
    :return:
    """
    c1 = (df.eq(1).sum(1) / df.shape[1]).gt(thr)
    c2 = (df.eq(2).sum(1) / df.shape[1]).gt(thr)
    return pd.DataFrame(np.select([c1, c2], [1, -1], 0), index=df.index, columns=['ensemble'])


if __name__ == '__main__':
    numDel = 0
    df = pd.read_csv("./Output/ensemble/results/walk0ensemble_test.csv", index_col='Date')

    fulldf = full_ensemble(df)

    for j in range(0, 5):
        df = pd.read_csv("./Output/ensemble/results/walk"+str(j)+"ensemble_test.csv", index_col='Date')

        for deleted in range(1, numDel):
            del df['iteration'+str(deleted)]

        fulldf = fulldf.append(full_ensemble(df))

    fulldf.to_csv("resultEnsembleTest.csv")
    fulldf = pd.read_csv("resultEnsembleTest.csv")
    fulldf['Date'] = pd.to_datetime(fulldf['Date'])
    fulldf = fulldf.set_index('Date')
    print(fulldf.head())

    fulldf.to_csv("spLong.csv")
