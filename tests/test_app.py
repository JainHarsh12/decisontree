import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

@pytest.fixture
def dataset():
    """Fixture for loading and splitting the Iris dataset."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    return train_test_split(X, y, test_size=0.3, random_state=42)

@pytest.fixture
def model():
    """Fixture for initializing the Decision Tree Classifier."""
    return DecisionTreeClassifier()

def test_model_training(model, dataset):
    """Test if the model trains successfully and the shape of predictions matches."""
    X_train, X_test, y_train, y_test = dataset
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert y_pred.shape == y_test.shape, "Prediction shape does not match target shape"

def test_model_accuracy(model, dataset):
    """Test if the model has acceptable accuracy."""
    X_train, X_test, y_train, y_test = dataset
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.8, f"Accuracy too low: {accuracy}"
