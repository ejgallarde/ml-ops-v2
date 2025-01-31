import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    """
    Preprocesses the dataset by checking for missing values and scaling continuous features.

    Args:
        data (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Preprocessed dataset.
        dict: Missing values count per feature.
    """
    # Check for missing values
    missing_values = data.isnull().sum()

    # Define continuous features to scale
    continuous_features = [
        'battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 
        'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 
        'ram', 'sc_h', 'sc_w', 'talk_time'
    ]

    # Scale the continuous features
    scaler = StandardScaler()
    data[continuous_features] = scaler.fit_transform(data[continuous_features])

    return data, missing_values
