"""
Module of functions to find customers who are likely to churn
Author: Gabriel da Paz
Date: November 2022
"""

# import libraries
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


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

    def perform_eda(self, df):
        '''
        perform eda on df and save figures to images folder
        input:
                df: pandas dataframe

        output:
                None
        '''

        plt.figure(figsize=(20, 10))
        df['Customer_Age'].hist()
        plt.savefig("./images/eda/customer_age_distribution.png")

        plt.figure(figsize=(20, 10))
        df.Marital_Status.value_counts(normalize=True).plot(kind='bar')
        plt.savefig("./images/eda/marital_status_distribution.png")

        plt.figure(figsize=(20, 10))
        df.plot.scatter(x='Customer_Age', y='Credit_Limit')
        plt.savefig("./images/eda/customer_age_credit_limit_relationship.png")

        plt.figure(figsize=(20, 10))
        numeric_types = [
            'int16',
            'int32',
            'int64',
            'float16',
            'float32',
            'float64']
        sns.heatmap(
            df.select_dtypes(
                include=numeric_types).corr(),
            annot=False,
            cmap='Dark2_r',
            linewidths=2)
        plt.savefig("./images/eda/corr_matrix.png")

    def encoder_helper(self, df, category_lst, response):
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook.
        Also create the Churn column from the Attrition_Flag column.

        input:
                df: pandas dataframe
                category_lst: list of columns that contain categorical features
                response: string of response name [optional argument that could be used for naming variables or index y column]

        output:
                df: pandas dataframe with new columns for
        '''
        df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
        

        for col in category_lst:
            new_col_vals = []
            groups = df.groupby(col).mean()[response]
            for val in df[col]:
                new_col_vals.append(groups.loc[val])
            df[col + "_" + response] = new_col_vals

        return df

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

    def classification_report_image(self,
                                    y_train,
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

    def feature_importance_plot(self, model, X_data, output_pth):
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

    def train_models(self, X_train, X_test, y_train, y_test):
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
    df = customer_churn.import_data("./data/bank_data.csv")
    customer_churn.perform_eda(df)
