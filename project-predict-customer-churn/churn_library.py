"""
Module of functions to find customers who are likely to churn
Author: Gabriel da Paz
Date: November 2022
"""

# import libraries
import pandas as pd

class CustomerChurn:
        """
        Pipeline with functions to find custoemrs who are likely to churn

        Attributes:

        path : str
                Path to the dataset that will be read

        """

        def __init__(self):
            pass

        def import_data(self, path):
                '''
                returns dataframe for the csv found at path

                input:
                        path: a path to the csv
                output:
                        df: pandas dataframe
                '''
                self.path = path
                df = pd.read_csv(path, index_col=0)

                return df


        def perform_eda(df):
                '''
                perform eda on df and save figures to images folder
                input:
                        df: pandas dataframe

                output:
                        None
                '''
                pass


        def encoder_helper(df, category_lst, response):
                '''
                helper function to turn each categorical column into a new column with
                propotion of churn for each category - associated with cell 15 from the notebook

                input:
                        df: pandas dataframe
                        category_lst: list of columns that contain categorical features
                        response: string of response name [optional argument that could be used for naming variables or index y column]

                output:
                        df: pandas dataframe with new columns for
                '''
                pass


        def perform_feature_engineering(df, response):
                '''
                input:
                        df: pandas dataframe
                        response: string of response name [optional argument that could be used for naming variables or index y column]

                output:
                        X_train: X training data
                        X_test: X testing data
                        y_train: y training data
                        y_test: y testing data
                '''

        def classification_report_image(y_train,
                                        y_test,
                                        y_train_preds_lr,
                                        y_train_preds_rf,
                                        y_test_preds_lr,
                                        y_test_preds_rf):
                '''
                produces classification report for training and testing results and stores report as image
                in images folder
                input:
                        y_train: training response values
                        y_test:  test response values
                        y_train_preds_lr: training predictions from logistic regression
                        y_train_preds_rf: training predictions from random forest
                        y_test_preds_lr: test predictions from logistic regression
                        y_test_preds_rf: test predictions from random forest

                output:
                        None
                '''
                pass


        def feature_importance_plot(model, X_data, output_pth):
                '''
                creates and stores the feature importances in pth
                input:
                        model: model object containing feature_importances_
                        X_data: pandas dataframe of X values
                        output_pth: path to store the figure

                output:
                        None
                '''
                pass

        def train_models(X_train, X_test, y_train, y_test):
                '''
                train, store model results: images + scores, and store models
                input:
                        X_train: X training data
                        X_test: X testing data
                        y_train: y training data
                        y_test: y testing data
                output:
                        None
                '''
                pass

if __name__ == "__main__":
        customer_churn = CustomerChurn()
