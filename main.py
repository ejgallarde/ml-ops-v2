from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.models.train_model import train_knn
from src.models.evaluate_model import evaluate_knn, plot_knn_accuracy
from src.models.predict import predict

if __name__ == "__main__":
    # Load the data
    data = load_data()

    # Preprocess the data
    preprocessed_data, missing_values = preprocess_data(data)
    print("Missing values per feature:")
    print(missing_values)

    # Build features and target
    X, y = build_features(preprocessed_data)
    print("Features (X):")
    print(X.head())
    print("Target (y):")
    print(y.head())

     # Train the model
    n_neighbors_range = range(1, 16)
    random_states = range(1, 21)
    scores_df = train_knn(X, y, n_neighbors_range, random_states)

    # Evaluate the model
    top_scores, mean_scores = evaluate_knn(scores_df)
    print("Top Scores:")
    print(top_scores)

    # Plot the results
    plot_knn_accuracy(mean_scores)

    # Make predictions
    knn = KNeighborsClassifier(n_neighbors=3)  # Example: Use the best value of n_neighbors
    knn.fit(X, y)
    new_data = X.sample(5)  # Example: Use a sample from X for prediction
    predictions = predict(knn, new_data)
    print("Predictions:", predictions)
