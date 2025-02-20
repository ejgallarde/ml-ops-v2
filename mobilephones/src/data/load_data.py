import os
import pandas as pd

def load_data(filename="data.csv"):
    """
    Loads a CSV file from the data folder.
    
    Args:
        filename (str): Name of the CSV file to load.
    
    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    # Get the absolute path to the project folder
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, "data", filename)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The file {filename} does not exist in the data folder.")
    
    return pd.read_csv(data_path)
