
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
    
def replace_nan_with_existing_val(df, column):
    """
    Replace NaN values in a DataFrame column with existing non-NaN values randomly.

    Parameters:
    - df: pandas DataFrame
    - column: str, the name of the column to process
    """
    # Get indices of NaN values in the specified column
    nan_indices = df.index[df[column].isna()].tolist()
    
    # Count the number of NaN values
    num_nans = len(nan_indices)
    
    # Check where values are not NaN, creating a replacement list
    value_index = df.index[df[column].notna()].tolist()
    replacement_list = df.loc[value_index, column].values.tolist()
    
    # Generate random values from the replacement list
    random_values = np.random.choice(replacement_list, size=num_nans)
    
    # Replace NaN values with randomly chosen values
    df.loc[nan_indices, column] = random_values
    
def preprocess_dataframe(numerical_features, categorical_features, df):
    """
    This function preprocesses a DataFrame by replacing NaN values and scaling numerical features.

    Parameters:
    - numerical_features: list, names of numerical columns
    - categorical_features: list, names of categorical columns
    - df: pandas DataFrame, the input DataFrame

    Returns:
    - X_preprocessed: numpy array, preprocessed feature matrix
    - feature_names: list, names of the output features after preprocessing
    """

    # Create transformers for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit and transform the data using the preprocessor
    X_preprocessed = preprocessor.fit_transform(df)

    # Get output feature names
    feature_names = preprocessor.get_feature_names_out()

    return X_preprocessed, feature_names

def extract_target_column(df,target_col):
    """
      function extracts target column from dataframe .
      converts it to integer and returns post flattening.
    """
    tran_y= df.pop(target_col)
    # type would be pandas.core.series.Series
    type(tran_y)
    y = pd.DataFrame(tran_y)
    y[target_col] = y[target_col].astype(int)
    y = y.values
    y = y.ravel()
    return y

def append_index(index, new_index):
    """
    Concatenate a new index with an existing index.

    Parameters:
    - index: pandas Index, the existing index
    - new_index: pandas Index, the index to be appended

    Returns:
    - combined_index: pandas Index, the combined index
    """
    # Concatenate the new index with the existing index
    combined_index = pd.Index(index.tolist() + new_index.tolist())
    
    # Return the combined index
    return combined_index

def dump_list_to_file(my_list, file_path):
    """
    Dump a list to a regular text file.

    Parameters:
    - my_list: list, the list to be dumped
    - file_path: str, the path to the output text file
    """
    # Dump the list to a text file
    with open(file_path, 'w') as file:
        for item in my_list:
            file.write(str(item) + '\n')
            
            
def plot_kde(data_column, df, title_name=None):
    """
    Visualize the frequency distribution using a KDE plot.

    Parameters:
    - data_column: str, the column to visualize
    - df: pandas DataFrame, the input DataFrame
    - title_name: str, the title for the plot (default is dynamically generated)

    Returns:
    None
    """
    # If title_name is not provided, generate a default title
    if title_name is None:
        title_name = f'Frequency Distribution of {data_column} KDE plot'

    # Visualize the frequency distribution using a KDE plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x=data_column, fill=True, color='skyblue', common_norm=False)
    plt.title(title_name)
    plt.xlabel(data_column)
    plt.ylabel('Density')
    plt.show()

def split_and_createnewcol(df):
    """
    Process the given DataFrame by splitting columns and converting data types.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame to be processed.

    Returns:
      None

    This function splits the 'PassengerId' and 'Cabin' columns, converts data types,
    and handles missing values to prepare the DataFrame for further analysis.
    """
    # Split the PassengerId into 2 columns: Group and NumInGroup
    df[['Group', 'NumInGroup']] = df["PassengerId"].str.split("_", expand=True)
    
    # Split the Cabin column into 3 columns: CabinDeck, CabinNum, and CabinSide
    df[['CabinDeck', 'CabinNum', 'CabinSide']] = df["Cabin"].str.split("/", expand=True)
    
    # Convert the 'Group' column to integer type
    df["Group"] = df["Group"].astype(int)
    
    # Replace NaN values in the "CabinNum" column with existing values
    replace_nan_with_existing_val(df, "CabinNum")
    
    # Convert the 'CabinNum' column to integer type to reduce the number of columns during preprocessing
    df["CabinNum"] = df["CabinNum"].astype(int)
    # check for NAN still exist, value of 0 indicates there is no NAN
    nan_len= len(df.index[df.CabinNum.isna()].tolist())
    assert nan_len == 0

    # drop the original column PassengerId and Cabin .
    # also drop name , does not have significance
    df.drop( ["Cabin", "Name", "PassengerId"], axis = 1 ,inplace=True)
    
def find_cat_num_features(df):
    """
    seperates out categorical and numerical features 
    Args:
        df (pandas.DataFrame): The input DataFrame
    Returns :
        numerical_features ( list)
        categorical_features (list)
    """
    numerical_features = df.select_dtypes(include=['float64','int32','int64']).columns
    print(f"numerical features {numerical_features}")
    categorical_features = df.select_dtypes(include=['object']).columns
    print(f"categorical features {categorical_features}")
    return ( numerical_features , categorical_features)