def build_features(data, target_column='price_range'):
    """
    Splits the dataset into features and target.

    Args:
        data (pd.DataFrame): Input dataset.
        target_column (str): The name of the target column.

    Returns:
        pd.DataFrame: Features (X).
        pd.Series: Target (y).
    """
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    return X, y
