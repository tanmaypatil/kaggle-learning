
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
from sklearn.metrics import roc_curve, auc

def add_num(a,b):
    return a + b

def replace_nan_frequent_val(df, column_name):
    """
    Replace NaN values with the most frequent value in the specified dataframe column.

    Parameters:
    - df: pandas DataFrame
    - column_name: str, the name of the column to process
    """
    # Replace NaN values with the most frequent value in the 'column_name' column
    most_frequent_value = df[column_name].mode()[0]
    df[column_name] = df[column_name].fillna(most_frequent_value)
