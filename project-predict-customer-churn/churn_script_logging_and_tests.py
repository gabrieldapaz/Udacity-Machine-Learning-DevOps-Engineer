"""
Module to test the functions of the package to
find customers who are likely to churn

Author: Gabriel da Paz
Date: November 2022
"""

# pylint: disable=no-member

import os
import logging
import math
import pytest
from churn_library import CustomerChurn

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        cc_obj = CustomerChurn()
        dataframe = cc_obj.import_data("./data/bank_data.csv")
        pytest.dataframe = dataframe
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    try:
        cc_obj = CustomerChurn()
        cc_obj.perform_eda(pytest.dataframe)

        cwd = os.path.abspath(os.getcwd())
        assert len(os.listdir(cwd + '/images/eda/')) == 4
        logging.info("Testing perform_eda: SUCCESS")
    except FileNotFoundError as err:
        logging.error(
            "Testing perform_eda: The directory doesn't appear to exist")
        raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''
    try:
        category_lst = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]
        response = 'Churn'

        cc_obj = CustomerChurn()
        dataframe = cc_obj.encoder_helper(
            pytest.dataframe,
            category_lst=category_lst,
            response=response)

        pytest.dataframe = dataframe

        df_select = dataframe[['Gender', 'Churn']]

        assert round(df_select.groupby('Gender').mean()[
            response].loc['M'], ndigits=4) == 0.1462
        assert round(df_select.groupby('Gender').mean()[
            response].loc['F'], ndigits=4) == 0.1736

        logging.info('Testing encoder helper: SUCCESS')

    except AssertionError as err:
        logging.error(
            "Testing encoder helper: The encoder doesn't appear to work properly"
        )
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    try:
        cc_obj = CustomerChurn()
        x_train, x_test, y_train, y_test = cc_obj.perform_feature_engineering(
            pytest.dataframe, response='Churn')

        assert x_train.shape[0] == math.floor(0.70 * pytest.dataframe.shape[0])
        assert y_train.shape[0] == math.floor(0.70 * pytest.dataframe.shape[0])
        assert x_test.shape[0] == math.ceil(0.30 * pytest.dataframe.shape[0])
        assert y_test.shape[0] == math.ceil(0.30 * pytest.dataframe.shape[0])

        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: \
                The train/test split doesn't appear to work properly")
        raise err


def test_train_models():
    '''
    test train_models
    '''
    try:
        cc_obj = CustomerChurn()
        x_train, x_test, y_train, y_test = cc_obj.perform_feature_engineering(
            pytest.dataframe, response='Churn')
        cc_obj.train_models(x_train, x_test, y_train, y_test)

        cwd = os.path.abspath(os.getcwd())
        assert len(os.listdir(cwd + '/images/results/')) == 7
        logging.info("Testing train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error(
            "Testing train_models: The directory doesn't appear to exist")
        raise err


if __name__ == "__main__":
    pass
