from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import joblib

data = fetch_olivetti_faces()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

model = joblib.load("savedmodel.pth")

accuracy = model.score(X_test, y_test)

print("Accuracy:", accuracy)
