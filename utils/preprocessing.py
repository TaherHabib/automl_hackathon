from typing import List

import numpy as np
import pandas as pd

import os

import config.settings

def convertCategoricalsToNumerics(df):
    """

    :param df:
    :return:
    """
    categorical_fields = df.select_dtypes(include=['object'])
    try:

        categorical_fields = categorical_fields.drop('class',axis=1)
    except(KeyError):
        print("Test data doesn't contain classes")
    for col in df.columns:
        if col in categorical_fields.columns:
            # Counting the different categories in the column
            countOrdered_cats = list(dict(categorical_fields[col].value_counts()).keys()) # descending order
            # Creating a list of numeric replacements
            num_replacements = list(np.arange(len(countOrdered_cats))+1)
            # Dictionary of replacements to pass to df.replace()
            replacements = dict(zip(countOrdered_cats, num_replacements))
            cd = df[col].replace(replacements, inplace=True)
    return df

def clean_data(dataframe:pd.DataFrame, columns_to_remove:List):
    """

    :return:
    """
    data = dataframe.drop(columns_to_remove, axis=1)
    data = data.fillna(0)
    data = convertCategoricalsToNumerics(data)

    return data

def save_csv(dataframe:pd.DataFrame,path:os.path):
    """

    :param dataframe:
    :param path:
    :return:
    """
    dataframe.to_csv(path)



if __name__ == '__main__':
    dataset = ['test']
    root = config.settings.get_project_path()
    dataset_path = os.path.join(root,'Data')

    for set in dataset:
        extension = '.csv'
        data_set = 'data_'+set+extension
        modify_set = set+extension
        set_df = pd.read_csv(os.path.join(dataset_path,data_set))
        column_name = ['V5', 'V23', 'V34', 'V44', 'V66', 'V86' ]
        data = clean_data(set_df,column_name)
        save_csv(data,os.path.join(dataset_path,modify_set))