import os
import logging
from churn_library import CustomerChurn
import pytest

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

class TestCustomerChurn:

    def test_import(self):
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
            logging.error("Testing import_data: The file doesn't appear to have rows and columns")
            raise err


    def test_eda(self):
        '''
        test perform eda function
        '''
        try:
            cc = CustomerChurn()
            cc.perform_eda(pytest.df)

            cwd = os.path.abspath(os.getcwd())
            assert len(os.listdir(cwd+'/images/eda/')) == 4
            logging.info("Testing perform_eda: SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing perform_eda: The directory doesn't appear to exist")
            raise err



    def test_encoder_helper(encoder_helper):
        '''
        test encoder helper
        '''


    def test_perform_feature_engineering(perform_feature_engineering):
        '''
        test perform_feature_engineering
        '''


    def test_train_models(train_models):
        '''
        test train_models
        '''


if __name__ == "__main__":
	pass








