# Predict Customer Churn

Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

The main goal of this project is to apply software engineering best practices to transform a jupyter notebook with data science code. The notebook contains code and analysis to identify customers more likely to churn.

This is a problem that most data science teams face in all kinds of situations. Imagine you work at a data science team that offers a product to identify customer churn. The data will change based on the client, but the process is quite similar. Instead of each data scientist creating a copy of this base notebook that isn't organized, it would be much more efficient and less prone to errors to use a well-tested package with periodic maintenance.

The data is available at [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/code).

## Files and data description

```
.
├── images               # Store EDA results
│   ├── eda
│   └── results
├── logs                 # Store logs
├── models               # Store models
├── README.md            # Provides project overview, and instructions to use the code
├── churn_library.py     # Define the functions (Was ToDo)
├── churn_notebook.ipynb # Contains the code to be refactored (Was given)
├── churn_script_logging_and_tests.py # Finish tests and logs (Was ToDo)
├── conftest.py
├── pytest.ini
├── requirements_py3.8.txt
└── data                 # Read this data (You will need to download from Kaggle)
    └── bank_data.csv

```

## Running Files

**1. Create a virtual environment**

The goal is to avoid dependency conflicts.
```
python3 -m venv /path/to/new/virtual/environment
```
You can check more details on virtual environments and how to activate them in this [link](https://docs.python.org/3/library/venv.html).

2. Install the requirements

```
pip3 install -r requirements_py3.8.txt
```

3. Run the code

You will need to modify the if __name__ == "__main__" part of the code to use the methods you want.

```
python3 churn_library.py
```

4. Test the code

After modifying the scripts is good practice to run the tests to check if everything is performing as expected.
You can check the results of the tests on the terminal and in the logs/ folder.

```
python3 churn_script_logging_and_tests.py
```

**Code Quality**

You can modify the code, but after is a good practice to ensure that the code is following the PEP8 standards.

Check code quality score with pylint.
```
python3 script.py
```

autopep8 automatically formats Python code to conform to the PEP 8 style guide.
```
autopep8 --in-place --aggressive --aggressive script.py
```
