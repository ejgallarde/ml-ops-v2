import pandas as pd
import matplotlib.pyplot as plt

def evaluate_knn(scores_df, training_threshold=0.90, test_threshold=0.80, top_n=20):
    """
    Evaluates the KNN model's performance and generates filtered scores.

    Args:
        scores_df (pd.DataFrame): DataFrame containing training and test scores.
        training_threshold (float): Minimum training accuracy for filtering.
        test_threshold (float): Minimum test accuracy for filtering.
        top_n (int): Number of top results to display.

    Returns:
        pd.DataFrame: Filtered and sorted scores DataFrame.
        pd.DataFrame: Mean scores grouped by n_neighbors.
    """
    # Filter the scores
    filtered_df = scores_df[(scores_df['training_accuracy'] >= training_threshold) & 
                            (scores_df['test_accuracy'] >= test_threshold)]

    # Calculate the difference between training and test accuracy
    filtered_df['accuracy_difference'] = abs(filtered_df['training_accuracy'] - filtered_df['test_accuracy'])

    # Sort by accuracy difference
    sorted_df = filtered_df.sort_values(by='accuracy_difference', ascending=True)

    # Group by n_neighbors and calculate mean scores
    mean_scores = scores_df.groupby('n_neighbors').agg({
        'training_accuracy': 'mean',
        'test_accuracy': 'mean'
    }).reset_index()

    return sorted_df.head(top_n), mean_scores

def plot_knn_accuracy(mean_scores):
    """
    Plots KNN accuracy vs. number of neighbors.

    Args:
        mean_scores (pd.DataFrame): DataFrame with mean scores grouped by n_neighbors.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(mean_scores['n_neighbors'], mean_scores['training_accuracy'], label='Training Accuracy', marker='o')
    plt.plot(mean_scores['n_neighbors'], mean_scores['test_accuracy'], label='Test Accuracy', marker='x')
    plt.title('KNN Accuracy vs. Number of Neighbors')
    plt.xlabel('Number of Neighbors (n_neighbors)')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, 16))
    plt.grid(True)
    plt.legend()
    plt.show()
