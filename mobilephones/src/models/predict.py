def predict(model, new_data):
    """
    Makes predictions using the trained model.

    Args:
        model (KNeighborsClassifier): Trained KNN model.
        new_data (pd.DataFrame): New data for prediction.

    Returns:
        np.array: Predictions.
    """
    return model.predict(new_data)
