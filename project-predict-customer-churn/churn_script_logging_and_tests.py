import os
import logging
from churn_library import CustomerChurn
import pytest

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
        cc = CustomerChurn()
        df = cc.import_data("./data/bank_data.csv")
        pytest.df = df
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    try:
        cc = CustomerChurn()
        cc.perform_eda(pytest.df)

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

        cc = CustomerChurn()
        df = cc.encoder_helper(
            pytest.df,
            category_lst=category_lst,
            response=response)

        pytest.df = df

        df_select = df[['Gender', 'Churn']]

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
        cc = CustomerChurn()
        X_train, X_test, y_train, y_test = cc.perform_feature_engineering(pytest.df)

        assert X_train.shape[0] == 0.70*pytest.df
        assert y_train.shape[0] == 0.70*pytest.df
        assert X_test.shape[0] == 0.30*pytest.df
        assert y_test.shape[0] == 0.30*pytest.df

        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The train/test split doesn't appear to work properly")
        raise err

def test_train_models():
    '''
    test train_models
    '''


if __name__ == "__main__":
    pass
