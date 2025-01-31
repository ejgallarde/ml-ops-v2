import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def train_knn(X, y, n_neighbors_range, random_states):
    """
    Trains a KNN model for a range of n_neighbors and random states.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        n_neighbors_range (range): Range of values for n_neighbors.
        random_states (range): Range of values for random states.

    Returns:
        pd.DataFrame: A DataFrame containing scores for all combinations of n_neighbors and random states.
    """
    scores = {
        'random_state': [],
        'n_neighbors': [],
        'training_accuracy': [],
        'test_accuracy': []
    }

    for random_state in random_states:
        for n_neighbors in n_neighbors_range:
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

            # Initialize and train the KNN model
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn.fit(X_train, y_train)

            # Evaluate the model
            y_train_pred = knn.predict(X_train)
            y_test_pred = knn.predict(X_test)
            training_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            # Store the scores
            scores['random_state'].append(random_state)
            scores['n_neighbors'].append(n_neighbors)
            scores['training_accuracy'].append(training_accuracy)
            scores['test_accuracy'].append(test_accuracy)

    return pd.DataFrame(scores)
