from dagster import asset, Definitions
from mobilephones.src.data.load_data import load_data
from mobilephones.src.data.preprocess import preprocess_data
from mobilephones.src.features.build_features import build_features
from mobilephones.src.models.train_model import train_knn
from mobilephones.src.models.evaluate_model import evaluate_knn
from mobilephones.src.models.predict import predict


asset_group_name = "mobilephones"

@asset(group_name = asset_group_name)
def raw_data():
    """Load raw data from the source."""
    return load_data()

@asset(group_name = asset_group_name)
def preprocessed_data(raw_data):
    """Preprocess the raw data."""
    return preprocess_data(raw_data)

@asset(group_name = asset_group_name)
def feature_data(preprocessed_data):
    """Perform feature engineering."""
    return build_features(preprocessed_data)

@asset(group_name = asset_group_name)
def trained_model(feature_data):
    """Train a machine learning model."""
    return train_knn(feature_data)

@asset(group_name = asset_group_name)
def evaluation_metrics(trained_model, feature_data):
    """Evaluate the trained model."""
    return evaluate_knn(trained_model, feature_data)

@asset(group_name = asset_group_name)
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