import pandas as pd

def load_data(filepath):
    """
    Loads the Iris dataset from a CSV file.
    """
    df = pd.read_csv(filepath)
    # If there's no 'Id' column, do not drop it
    return df
