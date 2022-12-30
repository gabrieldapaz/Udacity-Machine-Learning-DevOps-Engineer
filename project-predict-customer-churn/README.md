# Predict Customer Churn

Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

The main goal of this project is to apply software engineering best practices to transform a jupyter notebook with data science code. The notebook contains code and analysis to identify customers more likely to churn.

This is a problem that most data science teams face in all kinds of situations. Imagine you work at a data science team that offers a product to identify customer churn. The data will change based on the client, but the process is quite similar. Instead of each data scientist creating a copy of this base notebook that isn't organized, it would be much more efficient and less prone to errors to use a well-tested package with periodic maintenance.

The data is available at [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/code).

## Files and data description

```
.
├── churn_notebook.ipynb # Contains the code to be refactored (Was given)
├── churn_library.py     # Define the functions (Was ToDo)
├── churn_script_logging_and_tests.py # Finish tests and logs (Was ToDo)
├── README.md            # Provides project overview, and instructions to use the code
├── data                 # Read this data (You will need to download from Kaggle)
│   └── bank_data.csv
├── images               # Store EDA results 
│   ├── eda
│   └── results
├── logs                 # Store logs
└── models               # Store models

```

## Running Files

**1. Create a virtual environment**

The goal is to avoid dependency conflicts.
```
python3 -m venv /path/to/new/virtual/environment
```
You can check more details on virtual envs and how to activate them in this [link](https://docs.python.org/3/library/venv.html)

2. Install the requirements

```
pip3 install -r /path/to/requirements.txt
```
