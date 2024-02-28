import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression

X = np.loadtxt("datasets/x.txt")
y = np.loadtxt("datasets/y.txt")

model = LogisticRegression(random_state=0).fit(X, y)

joblib.dump(model, "models/sklearn_model.joblib")