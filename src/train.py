import joblib

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
model = LogisticRegression(random_state=0).fit(X, y)

joblib.dump(model, "../models/sklearn_model.joblib")