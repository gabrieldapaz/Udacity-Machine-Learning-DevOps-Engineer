"""
Module of functions to find customers who are likely to churn
Author: Gabriel da Paz
Date: November 2022
"""

# import libraries
import joblib
import shap
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import normalize
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

        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        for col in category_lst:
            new_col_vals = []
            groups = df.groupby(col).mean()[response]
            for val in df[col]:
                new_col_vals.append(groups.loc[val])
            df[col + "_" + response] = new_col_vals

        return df

    def perform_feature_engineering(self, df, response):
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

        category_lst = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]

        df = self.encoder_helper(
            df,
            category_lst=category_lst,
            response=response)

        y = df['Churn']
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

        X[keep_cols] = df[keep_cols]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        return X_train, X_test, y_train, y_test

    def _roc_curve_plot(self, y, pred, estimator_name, path_filename):
        """"
        Auxiliary function to plot and save roc curves
        """

        plt.figure(figsize=(20, 10))
        fpr, tpr, thresholds = roc_curve(y, pred)
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
        # scores
        # report_test_rf =  classification_report(y_test, y_test_preds_rf, output_dict=True)
        # report_train_rf = classification_report(y_train, y_train_preds_rf, output_dict=True)

        # report_test_lr = classification_report(y_test, y_test_preds_lr, output_dict=True)
        # report_train_lr = classification_report(y_train, y_train_preds_lr, output_dict=True)

        # with open('./results/classification_report.txt', 'w') as f:
        #     print('Train RF:', report_train_rf, file=f)
        #     print('Test RF:', report_test_rf, file=f)

        #     print('Train LR:', report_train_lr, file=f)
        #     print('Test LR:', report_test_lr, file=f)

        # f.close()

        plt.rc('figure', figsize=(5, 5))
        # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old
        # approach
        plt.text(0.01, 1.25, str('Random Forest Train'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
                 'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str('Random Forest Test'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
                 'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.axis('off')

        plt.rc('figure', figsize=(5, 5))
        plt.text(0.01, 1.25, str('Logistic Regression Train'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
                 'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str('Logistic Regression Test'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
                 'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.axis('off')

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
        cv_rfc.fit(X_train, y_train)

        lrc.fit(X_train, y_train)

        y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

        y_train_preds_lr = lrc.predict(X_train)
        y_test_preds_lr = lrc.predict(X_test)

        self.classification_report_image(
            y_train,
            y_test,
            y_train_preds_lr,
            y_train_preds_rf,
            y_test_preds_lr,
            y_test_preds_rf

        )

        # save best model
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == "__main__":
    customer_churn = CustomerChurn()
    df = customer_churn.import_data("./data/bank_data.csv")
    customer_churn.perform_eda(df)
    X_train, X_test, y_train, y_test = customer_churn.perform_feature_engineering(
        df, response='Churn')
    customer_churn.train_models(X_train, X_test, y_train, y_test)
