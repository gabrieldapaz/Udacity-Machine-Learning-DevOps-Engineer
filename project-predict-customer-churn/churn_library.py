"""
Module of functions to find customers who are likely to churn
Author: Gabriel da Paz
Date: November 2022
"""

# import libraries
import joblib
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import RocCurveDisplay, classification_report, roc_curve, auc


class CustomerChurn:
    """
    Pipeline with functions to find customers who are likely to churn

    Attributes:

    path : str
            Path to the dataset that will be read

    """

    def __init__(self):
        self.path = None

    def import_data(self, path):
        '''
        returns dataframe for the csv found at path

        input:
                path: a path to the csv
        output:
                dataframe: pandas dataframe
        '''
        self.path = path
        dataframe = pd.read_csv(path, index_col=0)

        return dataframe

    def perform_eda(self, dataframe):
        '''
        perform eda on dataframe and save figures to images folder
        input:
                dataframe: pandas dataframe

        output:
                None
        '''

        plt.figure(figsize=(20, 10))
        dataframe['Customer_Age'].hist()
        plt.savefig("./images/eda/customer_age_distribution.png")

        plt.figure(figsize=(20, 10))
        dataframe.Marital_Status.value_counts(normalize=True).plot(kind='bar')
        plt.savefig("./images/eda/marital_status_distribution.png")

        plt.figure(figsize=(20, 10))
        dataframe.plot.scatter(x='Customer_Age', y='Credit_Limit')
        plt.savefig("./images/eda/customer_age_credit_limit_relationship.png")

        plt.figure(figsize=(20, 15))
        numeric_types = [
            'int16',
            'int32',
            'int64',
            'float16',
            'float32',
            'float64']
        sns.heatmap(
            dataframe.select_dtypes(
                include=numeric_types).corr(),
            annot=False,
            cmap='Dark2_r',
            linewidths=2)
        plt.savefig("./images/eda/corr_matrix.png")

    def encoder_helper(self, dataframe, category_lst, response):
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook.
        Also create the Churn column from the Attrition_Flag column.

        input:
                dataframe: pandas dataframe
                category_lst: list of columns that contain categorical features
                response: string of response name [optional argument that could be used
                          for naming variables or index y column]

        output:
                dataframe: pandas dataframe with new columns for
        '''

        dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        for col in category_lst:
            new_col_vals = []
            groups = dataframe.groupby(col).mean()[response]
            for val in dataframe[col]:
                new_col_vals.append(groups.loc[val])
            dataframe[col + "_" + response] = new_col_vals

        return dataframe

    def perform_feature_engineering(self, dataframe, response):
        '''
        input:
                dataframe: pandas dataframe
                response: string of response name [optional argument that could
                be used for naming variables or index y column]

        output:
                x_train: X training data
                x_test: X testing data
                y_train: y training data
                y_test: y testing data
        '''

        category_lst = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]

        dataframe = self.encoder_helper(
            dataframe,
            category_lst=category_lst,
            response=response)

        y = dataframe['Churn']
        X = pd.DataFrame()

        keep_cols = [
            'Customer_Age',
            'Dependent_count',
            'Months_on_book',
            'Total_Relationship_Count',
            'Months_Inactive_12_mon',
            'Contacts_Count_12_mon',
            'Credit_Limit',
            'Total_Revolving_Bal',
            'Avg_Open_To_Buy',
            'Total_Amt_Chng_Q4_Q1',
            'Total_Trans_Amt',
            'Total_Trans_Ct',
            'Total_Ct_Chng_Q4_Q1',
            'Avg_Utilization_Ratio',
            'Gender_Churn',
            'Education_Level_Churn',
            'Marital_Status_Churn',
            'Income_Category_Churn',
            'Card_Category_Churn']

        X[keep_cols] = dataframe[keep_cols]

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        return x_train, x_test, y_train, y_test

    def _roc_curve_plot(self, y, pred, estimator_name, path_filename):
        """
        Auxiliary function to plot and save roc curves
        input:
                y: list of ground truth
                pred: list of prediction
                estimator_name: estimator name
                path_filename: path where the images will be saved

        output:
                None
        """
        plt.clf()
        plt.figure(figsize=(20, 10))
        fpr, tpr, _ = roc_curve(y, pred)
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name=estimator_name)
        display.plot()
        plt.savefig(path_filename)

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

        plt.clf()
        plt.rc('figure', figsize=(5, 5))
        # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old
        # approach
        plt.text(0.01, 1.25, str('Random Forest Train'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.6, str('Random Forest Test'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig("./images/results/classification_report_rf.png")

        plt.clf()
        plt.rc('figure', figsize=(5, 5))
        plt.text(0.01, 1.25, str('Logistic Regression Train'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.6, str('Logistic Regression Test'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig("./images/results/classification_report_lr.png")

        self._roc_curve_plot(
            y_test,
            y_test_preds_rf,
            estimator_name="Random Forest",
            path_filename="./images/results/roc_curve_rf_test.png")

        self._roc_curve_plot(
            y_train,
            y_train_preds_rf,
            estimator_name="Random Forest",
            path_filename="./images/results/roc_curve_rf_train.png")

        self._roc_curve_plot(
            y_test,
            y_test_preds_lr,
            estimator_name="Logistic Regression",
            path_filename="./images/results/roc_curve_lr_test.png")

        self._roc_curve_plot(
            y_train,
            y_train_preds_lr,
            estimator_name="Logistic Regression",
            path_filename="./images/results/roc_curve_lr_train.png")

    def feature_importance_plot(self, model, x_data, output_pth):
        '''
        creates and stores the feature importances in pth
        input:
                model: model object containing feature_importances_
                x_data: pandas dataframe of X values
                output_pth: path to store the figure

        output:
                None
        '''
        # Calculate feature importances
        importances = model.best_estimator_.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [x_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 10))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(x_data.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(x_data.shape[1]), names, rotation=90)

        plt.savefig(output_pth)

    def train_models(self, x_train, x_test, y_train, y_test):
        '''
        train, store model results: images + scores, and store models
        input:
                x_train: X training data
                x_test: X testing data
                y_train: y training data
                y_test: y testing data
        output:
                None
        '''
        # grid search
        rfc = RandomForestClassifier(random_state=42)
        # Use a different solver if the default 'lbfgs' fails to converge
        # Reference:
        # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(x_train, y_train)

        lrc.fit(x_train, y_train)

        y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

        y_train_preds_lr = lrc.predict(x_train)
        y_test_preds_lr = lrc.predict(x_test)

        self.classification_report_image(
            y_train,
            y_test,
            y_train_preds_lr,
            y_train_preds_rf,
            y_test_preds_lr,
            y_test_preds_rf

        )

        self.feature_importance_plot(
            cv_rfc, x_train, "./images/results/feature_importance_rf.png")

        # save best model
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == "__main__":
    pass
    # customer_churn = CustomerChurn()
    # dataframe = customer_churn.import_data("./data/bank_data.csv")
    # customer_churn.perform_eda(dataframe)
    # x_train, x_test, y_train, y_test = customer_churn.perform_feature_engineering(
    #     dataframe, response='Churn')
    # customer_churn.train_models(x_train, x_test, y_train, y_test)
