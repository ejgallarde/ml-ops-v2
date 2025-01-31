from dagster import asset
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate_model
from src.models.predict import predict

@asset
def raw_data():
    """Load raw data from the source."""
    return load_data()

@asset
def preprocessed_data(raw_data):
    """Preprocess the raw data."""
    return preprocess_data(raw_data)

@asset
def feature_data(preprocessed_data):
    """Perform feature engineering."""
    return build_features(preprocessed_data)

@asset
def trained_model(feature_data):
    """Train a machine learning model."""
    return train_model(feature_data)

@asset
def evaluation_metrics(trained_model, feature_data):
    """Evaluate the trained model."""
    return evaluate_model(trained_model, feature_data)

@asset
def predictions(trained_model, feature_data):
    """Make predictions using the trained model."""
    return predict(trained_model, feature_data)

# Register assets in Definitions
definitions = Definitions(assets=[
    raw_data,
    preprocessed_data,
    feature_data,
    trained_model,
    evaluation_metrics,
    predictions
])